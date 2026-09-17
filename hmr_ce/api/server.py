"""
FastAPI Server and WebSocket Telemetry for HMR-CE
Exposes REST endpoints and live WebSocket streaming for the interactive visual dashboard.
"""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

from hmr_ce.config import HMRCEConfig, default_config
from hmr_ce.schemas import SpeakerRole, ActiveStatus
from hmr_ce.engine.coordinator import HMRCECoordinator
from hmr_ce.agent.chat_agent import HMRCEChatAgent


class ChatRequest(BaseModel):
    message: str
    force_verbatim: bool = False
    is_retrospective: Optional[bool] = None


class IngestRequest(BaseModel):
    text: str
    speaker: SpeakerRole = SpeakerRole.USER


class RetrieveRequest(BaseModel):
    query: str
    force_verbatim: bool = False
    is_retrospective: Optional[bool] = None


class ConnectionManager:
    def __init__(self):
        self.active_connections: List[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

    async def broadcast(self, message: Dict[str, Any]):
        msg_str = json.dumps(message)
        for connection in list(self.active_connections):
            try:
                await connection.send_text(msg_str)
            except Exception:
                self.disconnect(connection)


def create_app(config: Optional[HMRCEConfig] = None) -> FastAPI:
    cfg = config or default_config
    coordinator = HMRCECoordinator(cfg)
    agent = HMRCEChatAgent(coordinator, cfg)
    ws_manager = ConnectionManager()

    app = FastAPI(title="HMR-CE Context Engine", version="1.0.0")

    ui_dir = Path(__file__).resolve().parent.parent / "ui"
    static_dir = ui_dir / "static"
    static_dir.mkdir(parents=True, exist_ok=True)

    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

    @app.get("/", response_class=HTMLResponse)
    async def serve_ui():
        index_path = ui_dir / "index.html"
        if not index_path.exists():
            return "<h3>HMR-CE Dashboard UI is initializing...</h3>"
        return FileResponse(str(index_path))

    @app.get("/api/state")
    async def get_state():
        return coordinator.get_full_topology_state()

    @app.get("/api/graph")
    async def get_graph():
        """Returns nodes and directed edges for belief DAG visualizer."""
        nodes = coordinator.tier2_store.get_all_nodes()
        t1_turns = {t.turn_id: t for t in coordinator.tier1_store.get_all_turns(coordinator.session_id)}
        
        graph_nodes = []
        graph_edges = []

        for n in nodes:
            turn_id = n.turn_references[-1] if n.turn_references else 0
            t1_rec = t1_turns.get(turn_id)
            
            # The Belief Revision Graph represents the agent's world model of USER assertions & retractions.
            # Filter out assistant conversational pleasantries / acknowledgments from the graph.
            is_assistant = False
            if t1_rec:
                role_val = getattr(t1_rec.speaker_role, "value", str(t1_rec.speaker_role))
                if str(role_val).lower() == "assistant":
                    is_assistant = True
            if not is_assistant and n.metadata and str(n.metadata.get("speaker", "")).lower() == "assistant":
                is_assistant = True

            if is_assistant:
                continue

            raw_text = t1_rec.raw_text if t1_rec else n.metadata.get("abstract", "")
            
            graph_nodes.append({
                "id": n.node_id,
                "label": f"Turn {turn_id}: {raw_text[:28]}..." if len(raw_text) > 28 else f"Turn {turn_id}: {raw_text}",
                "full_text": raw_text,
                "status": n.status.value,
                "salience": n.surprise_salience,
                "cluster_id": n.centroid_cluster_id,
                "momentum_inherited": n.momentum_inherited,
                "turn_id": turn_id,
                "superseded_by": n.superseded_by_node
            })

        node_id_set = {gn["id"] for gn in graph_nodes}
        for n in nodes:
            if n.superseded_by_node and n.node_id in node_id_set and n.superseded_by_node in node_id_set:
                graph_edges.append({
                    "from": n.node_id,
                    "to": n.superseded_by_node,
                    "type": "SUPERSEDED_BY",
                    "color": "#ef4444"
                })

        return {"nodes": graph_nodes, "edges": graph_edges}

    @app.post("/api/chat")
    async def chat(req: ChatRequest):
        res = agent.chat_turn(
            user_message=req.message,
            force_verbatim=req.force_verbatim,
            is_retrospective=req.is_retrospective
        )
        await ws_manager.broadcast({
            "event": "turn_processed",
            "user_ingest": res["user_ingest"],
            "assistant_ingest": res["assistant_ingest"]
        })
        return res

    @app.post("/api/ingest")
    async def ingest(req: IngestRequest):
        res = coordinator.ingest_turn(text=req.text, speaker=req.speaker)
        res_dict = res.model_dump()
        await ws_manager.broadcast({"event": "turn_ingested", "data": res_dict})
        return res_dict

    @app.post("/api/retrieve")
    async def retrieve(req: RetrieveRequest):
        res = coordinator.retrieve(
            query=req.query,
            force_verbatim=req.force_verbatim,
            is_retrospective=req.is_retrospective
        )
        return res.model_dump()

    @app.post("/api/reset")
    async def reset():
        coordinator.reset_session()
        await ws_manager.broadcast({"event": "session_reset"})
        return {"status": "success", "message": "Memory session reset"}

    @app.post("/api/scenarios/{scenario_id}")
    async def run_scenario(scenario_id: str):
        """Execute one of the 5 Section 7 failure mode scenarios."""
        coordinator.reset_session()

        if scenario_id == "belief_invalidation":
            # 1. Proposition
            t1 = coordinator.ingest_turn("Let's implement our message queue using DNS TXT records.", SpeakerRole.USER)
            # 2. Explanation under Titans momentum window
            t2 = coordinator.ingest_turn("We can configure the DNS TTL to 0 seconds to prevent stale caching.", SpeakerRole.USER)
            # 3. Intermediate turns
            t3 = coordinator.ingest_turn("What about the client polling interval?", SpeakerRole.ASSISTANT)
            t4 = coordinator.ingest_turn("Let's keep polling at 100 milliseconds.", SpeakerRole.USER)
            # 4. Refutation / Contradiction override
            t5 = coordinator.ingest_turn("Actually, DNS TXT failed due to ISP caching; switch to Redis Streams instead.", SpeakerRole.USER)

            # Standard query (should retrieve Redis, suppress DNS)
            q_std = coordinator.retrieve("What queue architecture are we using?")
            # Retrospective query (should retrieve superseded DNS node)
            q_retro = coordinator.retrieve("Why did we decide against DNS?", is_retrospective=True)

            return {
                "scenario": "Belief Invalidation & DAG Revision",
                "turns_ingested": 5,
                "turns": [
                    {"text": t1.tier1_record.raw_text, "gate_result": t1.gate_result.model_dump()},
                    {"text": t2.tier1_record.raw_text, "gate_result": t2.gate_result.model_dump()},
                    {"text": t4.tier1_record.raw_text, "gate_result": t4.gate_result.model_dump()},
                    {"text": t5.tier1_record.raw_text, "gate_result": t5.gate_result.model_dump()},
                ],
                "invalidation_detected": len(t5.invalidations) > 0,
                "standard_retrieval_active_only": [n.node_id for n in q_std.scored_nodes],
                "retrospective_retrieval_includes_superseded": [n.node_id for n in q_retro.scored_nodes if n.status == ActiveStatus.SUPERSEDED],
                "topology": coordinator.get_full_topology_state()
            }

        elif scenario_id == "semantic_smearing":
            # Baseline domain turns
            t1 = coordinator.ingest_turn("We are optimizing PostgreSQL query plans.", SpeakerRole.USER)
            t2 = coordinator.ingest_turn("The B-tree index on user_id improves lookup speed.", SpeakerRole.USER)
            # Quadrant 2 Novel proposition
            q2_turn = coordinator.ingest_turn("Store full binary serialized tensors directly inside JSONB column.", SpeakerRole.USER)
            
            return {
                "scenario": "Semantic Smearing Prevention via Surprise Isolation",
                "turns": [
                    {"text": t1.tier1_record.raw_text, "gate_result": t1.gate_result.model_dump()},
                    {"text": t2.tier1_record.raw_text, "gate_result": t2.gate_result.model_dump()},
                    {"text": q2_turn.tier1_record.raw_text, "gate_result": q2_turn.gate_result.model_dump()},
                ],
                "q2_quadrant": q2_turn.gate_result.quadrant.value,
                "q2_is_isolated": q2_turn.gate_result.is_isolated,
                "salience": q2_turn.gate_result.salience,
                "topology": coordinator.get_full_topology_state()
            }

        elif scenario_id == "titans_momentum":
            # Trigger novelty turn
            trig = coordinator.ingest_turn("Let's implement our message queue using DNS TXT records.", SpeakerRole.USER)
            # Follow-up turns
            f1 = coordinator.ingest_turn("DNS resolvers provide global edge distribution.", SpeakerRole.USER)
            f2 = coordinator.ingest_turn("The payload can be base64 encoded into multiple TXT strings.", SpeakerRole.USER)
            f3 = coordinator.ingest_turn("Edge nodes will execute anycast routing.", SpeakerRole.USER)

            return {
                "scenario": "Titans Forward Momentum Window",
                "turns": [
                    {"text": trig.tier1_record.raw_text, "gate_result": trig.gate_result.model_dump()},
                    {"text": f1.tier1_record.raw_text, "gate_result": f1.gate_result.model_dump()},
                    {"text": f2.tier1_record.raw_text, "gate_result": f2.gate_result.model_dump()},
                    {"text": f3.tier1_record.raw_text, "gate_result": f3.gate_result.model_dump()},
                ],
                "trigger_salience": trig.gate_result.salience,
                "followup_1_salience": f1.gate_result.salience,
                "followup_2_salience": f2.gate_result.salience,
                "followup_3_salience": f3.gate_result.salience,
                "active_momentum": coordinator.surprise_gate.active_momentum,
                "topology": coordinator.get_full_topology_state()
            }

        elif scenario_id == "temporal_inversion":
            t1 = coordinator.ingest_turn("Initial server IP is 192.168.1.50", SpeakerRole.USER)
            middle_turns = []
            for i in range(12):
                mt = coordinator.ingest_turn(f"Routine system check {i}: all healthchecks green.", SpeakerRole.ASSISTANT)
                if i in (0, 6, 11):
                    middle_turns.append(mt)
            t_recent = coordinator.ingest_turn("Server IP updated to 10.0.0.1", SpeakerRole.USER)

            q = coordinator.retrieve("What is the server IP address?")
            turns_list = [
                {"text": t1.tier1_record.raw_text, "gate_result": t1.gate_result.model_dump()}
            ] + [
                {"text": mt.tier1_record.raw_text, "gate_result": mt.gate_result.model_dump()} for mt in middle_turns
            ] + [
                {"text": t_recent.tier1_record.raw_text, "gate_result": t_recent.gate_result.model_dump()}
            ]
            return {
                "scenario": "Temporal Inversion Prevention via Exponential Decay",
                "turns": turns_list,
                "top_retrieved_turn": q.scored_nodes[0].turn_references if q.scored_nodes else [],
                "scores": [(n.turn_references, n.composite_score, n.time_decay) for n in q.scored_nodes],
                "retrieval": q.model_dump(),
                "topology": coordinator.get_full_topology_state()
            }

        elif scenario_id == "prefix_truncation":
            t1 = coordinator.ingest_turn("Machine learning transformers and quadratic attention.", SpeakerRole.USER)
            t2 = coordinator.ingest_turn("PostgreSQL indexing and relational databases.", SpeakerRole.USER)
            t3 = coordinator.ingest_turn("Frontend React components and CSS grid styling.", SpeakerRole.USER)

            q = coordinator.retrieve("How do relational databases optimize queries?")
            return {
                "scenario": "Two-Stage Funnel (MRL d=64 Coarse Sweep -> d=1024 Fine Re-Rank)",
                "turns": [
                    {"text": t1.tier1_record.raw_text, "gate_result": t1.gate_result.model_dump()},
                    {"text": t2.tier1_record.raw_text, "gate_result": t2.gate_result.model_dump()},
                    {"text": t3.tier1_record.raw_text, "gate_result": t3.gate_result.model_dump()},
                ],
                "candidate_clusters": q.candidate_clusters,
                "scored_nodes": [(n.turn_references, n.sim_fine, n.composite_score) for n in q.scored_nodes],
                "retrieval": q.model_dump(),
                "topology": coordinator.get_full_topology_state()
            }

        raise HTTPException(status_code=404, detail="Scenario not found")

    @app.websocket("/ws/stream")
    async def websocket_endpoint(websocket: WebSocket):
        await ws_manager.connect(websocket)
        try:
            while True:
                data = await websocket.receive_text()
                # Echo ping
                await websocket.send_text(json.dumps({"event": "pong", "data": data}))
        except WebSocketDisconnect:
            ws_manager.disconnect(websocket)

    return app


app = create_app()
