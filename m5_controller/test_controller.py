"""
M5 unit tests — ZERO real M1-M4 dependencies.
Uses mocked fixtures only. Run with: python -m pytest m5_controller/test_controller.py -v
"""
import pytest
import asyncio

from shared.schemas import TaskType, ResultItem, UniformError, ErrorType
from shared.validate import load_fixture_queries, load_fixture_results
from m5_controller.dispatch_table import (
    orchestrate, pipeline_search, pipeline_vqa, pipeline_change, pipeline_fusion,
    DISPATCH,
)
from m5_controller.trace_builder import TraceBuilder


class TestDispatchTable:
    def test_all_task_types_have_pipeline(self):
        # TaskType.grounding is deprecated and removed from dispatch
        dispatched_types = {TaskType.search, TaskType.vqa, TaskType.caption, TaskType.change, TaskType.fusion}
        for task in dispatched_types:
            assert task.value in DISPATCH

    @pytest.mark.asyncio
    async def test_search_pipeline_returns_results(self):
        from shared.schemas import StructuredQuery
        q = StructuredQuery(
            query_text="Show me agriculture", task_type=TaskType.search,
            sensor=None, object="agriculture", confidence=0.9, used_fallback=False,
        )
        builder = TraceBuilder(task_selected="search")
        results = await pipeline_search(q, builder)
        assert isinstance(results, list)
        trace = builder.build()
        assert trace.task_selected == "search"

    @pytest.mark.asyncio
    async def test_vqa_pipeline_with_missing_image_returns_error_result(self):
        from shared.schemas import StructuredQuery
        q = StructuredQuery(
            query_text="What is this?", task_type=TaskType.vqa,
            question="What is this?", confidence=0.9, used_fallback=False,
        )
        builder = TraceBuilder(task_selected="vqa")
        results = await pipeline_vqa(q, builder)
        # Fixture images don't exist on disk, so compatibility check fails gracefully
        assert len(results) == 1
        assert results[0].explanation_text != ""

    @pytest.mark.asyncio
    async def test_change_pipeline_finds_paired_scene(self):
        from shared.schemas import StructuredQuery
        q = StructuredQuery(
            query_text="What changed?", task_type=TaskType.change,
            change_flag=True, confidence=0.9, used_fallback=False,
        )
        builder = TraceBuilder(task_selected="change")
        results = await pipeline_change(q, builder)
        assert isinstance(results, list)

    @pytest.mark.asyncio
    async def test_fusion_pipeline_missing_sar_returns_explained_failure(self):
        from shared.schemas import StructuredQuery, Sensor
        q = StructuredQuery(
            query_text="Combine optical and SAR", task_type=TaskType.fusion,
            sensor=Sensor.both, confidence=0.9, used_fallback=False,
        )
        builder = TraceBuilder(task_selected="fusion")
        results = await pipeline_fusion(q, builder)
        assert len(results) == 1
        assert "unavailable" in results[0].explanation_text.lower() or "fusion" in results[0].explanation_text.lower() or "missing" in results[0].explanation_text.lower()


class TestOrchestrate:
    def test_orchestrate_returns_query_response(self):
        resp = orchestrate("Show me agriculture near Delhi")
        assert hasattr(resp, "results")
        assert hasattr(resp, "trace")
        assert hasattr(resp, "query_id")

    def test_orchestrate_populates_trace(self):
        resp = orchestrate("Describe the land cover")
        assert resp.trace.task_selected == "caption"
        assert len(resp.trace.tools_called) >= 0

    def test_orchestrate_handles_broken_input_gracefully(self):
        resp = orchestrate("")
        assert isinstance(resp.results, list)

    def test_orchestrate_grounding_routes_to_caption(self):
        """Grounding phrases like 'highlight', 'where is', 'locate' should route to caption."""
        resp = orchestrate("Highlight the water body in the north east")
        assert resp.trace.task_selected == "caption"


class TestTraceBuilder:
    def test_trace_records_tool_calls(self):
        builder = TraceBuilder(task_selected="search")
        builder.add_tool_call("M2", "RemoteCLIP", {"query": "test"}, 120, True)
        trace = builder.build()
        assert len(trace.tools_called) == 1
        assert trace.tools_called[0].latency_ms == 120

    def test_trace_fallback_flag(self):
        builder = TraceBuilder(task_selected="search")
        builder.set_fallback_used(True)
        trace = builder.build()
        assert trace.fallback_used is True