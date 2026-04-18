from __future__ import annotations

from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any


class AirflowException(Exception):
    pass


class TriggerRule:
    ALL_DONE = "all_done"
    ALL_SUCCESS = "all_success"


_DAG_STACK: list["DAG"] = []
_TASK_GROUP_STACK: list["TaskGroup"] = []


def _iter_nodes(value: Any) -> list["_DependencyNode"]:
    if isinstance(value, (list, tuple, set)):
        nodes: list[_DependencyNode] = []
        for item in value:
            nodes.extend(_iter_nodes(item))
        return nodes
    if isinstance(value, TaskGroup):
        return list(value.root_tasks)
    if isinstance(value, _DependencyNode):
        return [value]
    raise TypeError(f"Unsupported dependency target: {type(value)!r}")


class _DependencyNode:
    task_id: str
    upstream_task_ids: set[str]
    downstream_task_ids: set[str]

    def set_downstream(self, other: Any) -> Any:
        targets = _iter_nodes(other)
        for target in targets:
            self.downstream_task_ids.add(target.task_id)
            target.upstream_task_ids.add(self.task_id)
        return other

    def set_upstream(self, other: Any) -> Any:
        sources = _iter_nodes(other)
        for source in sources:
            source.downstream_task_ids.add(self.task_id)
            self.upstream_task_ids.add(source.task_id)
        return other

    def __rshift__(self, other: Any) -> Any:
        return self.set_downstream(other)

    def __lshift__(self, other: Any) -> Any:
        return self.set_upstream(other)

    def __rrshift__(self, other: Any) -> "_DependencyNode":
        self.set_upstream(other)
        return self

    def __rlshift__(self, other: Any) -> "_DependencyNode":
        self.set_downstream(other)
        return self


class DAG(AbstractContextManager):
    def __init__(self, dag_id: str, **kwargs: Any) -> None:
        self.dag_id = dag_id
        self.kwargs = kwargs
        self._tasks: dict[str, _DependencyNode] = {}

    def __enter__(self) -> "DAG":
        _DAG_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _DAG_STACK and _DAG_STACK[-1] is self:
            _DAG_STACK.pop()
        return None

    def add_task(self, task: _DependencyNode) -> None:
        self._tasks[task.task_id] = task

    @property
    def task_ids(self) -> list[str]:
        return list(self._tasks.keys())

    def get_task(self, task_id: str) -> _DependencyNode:
        return self._tasks[task_id]


class TaskGroup(AbstractContextManager):
    def __init__(self, group_id: str, dag: DAG | None = None, **kwargs: Any) -> None:
        self.group_id = group_id
        self.dag = dag or (_DAG_STACK[-1] if _DAG_STACK else None)
        self.kwargs = kwargs
        self.tasks: list[_DependencyNode] = []

    def __enter__(self) -> "TaskGroup":
        _TASK_GROUP_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if _TASK_GROUP_STACK and _TASK_GROUP_STACK[-1] is self:
            _TASK_GROUP_STACK.pop()
        return None

    def add_task(self, task: _DependencyNode) -> None:
        self.tasks.append(task)

    @property
    def root_tasks(self) -> list[_DependencyNode]:
        task_ids = {task.task_id for task in self.tasks}
        return [task for task in self.tasks if not (task.upstream_task_ids & task_ids)]

    @property
    def leaf_tasks(self) -> list[_DependencyNode]:
        task_ids = {task.task_id for task in self.tasks}
        return [task for task in self.tasks if not (task.downstream_task_ids & task_ids)]

    def __rshift__(self, other: Any) -> Any:
        for leaf in self.leaf_tasks:
            leaf.set_downstream(other)
        return other

    def __lshift__(self, other: Any) -> Any:
        for root in self.root_tasks:
            root.set_upstream(other)
        return other

    def __rrshift__(self, other: Any) -> "TaskGroup":
        for root in self.root_tasks:
            root.set_upstream(other)
        return self

    def __rlshift__(self, other: Any) -> "TaskGroup":
        for leaf in self.leaf_tasks:
            leaf.set_downstream(other)
        return self


@dataclass
class PythonOperator(_DependencyNode):
    task_id: str
    python_callable: Any
    dag: DAG | None = None
    trigger_rule: str = TriggerRule.ALL_SUCCESS

    def __post_init__(self) -> None:
        self.upstream_task_ids = set()
        self.downstream_task_ids = set()
        current_group = _TASK_GROUP_STACK[-1] if _TASK_GROUP_STACK else None
        if current_group is not None:
            self.task_id = f"{current_group.group_id}.{self.task_id}"
            current_group.add_task(self)
        current_dag = self.dag or (_DAG_STACK[-1] if _DAG_STACK else None)
        if current_dag is not None:
            current_dag.add_task(self)
