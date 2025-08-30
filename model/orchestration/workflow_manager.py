"""
Workflow Manager
===============

Manages complex workflows and multi-step processes in the BreadthFlow system.
Handles workflow definitions, scheduling, execution, and monitoring.
"""

import asyncio
import json
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from ..logging.enhanced_logger import EnhancedLogger
from ..logging.error_handler import ErrorHandler
from ..logging.error_recovery import ErrorRecovery


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepStatus(Enum):
    """Individual step execution status"""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass
class WorkflowStep:
    """Definition of a workflow step"""
    step_id: str
    name: str
    description: str
    function: Callable
    dependencies: List[str] = field(default_factory=list)
    timeout_seconds: int = 300
    retry_attempts: int = 3
    retry_delay_seconds: int = 60
    required: bool = True
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowExecution:
    """Workflow execution record"""
    execution_id: str
    workflow_id: str
    status: WorkflowStatus
    start_time: datetime
    end_time: Optional[datetime] = None
    steps: Dict[str, StepStatus] = field(default_factory=dict)
    results: Dict[str, Any] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowDefinition:
    """Complete workflow definition"""
    workflow_id: str
    name: str
    description: str
    version: str
    steps: List[WorkflowStep]
    max_concurrent_executions: int = 1
    timeout_seconds: int = 3600
    tags: List[str] = field(default_factory=list)


class WorkflowManager:
    """
    Manages workflow definitions and executions
    
    Features:
    - Workflow definition and validation
    - Dependency resolution and execution ordering
    - Parallel and sequential execution
    - Error handling and retry logic
    - Execution monitoring and history
    - Workflow scheduling
    """
    
    def __init__(self):
        """Initialize the workflow manager"""
        self.logger = EnhancedLogger("WorkflowManager")
        self.error_handler = ErrorHandler()
        
        # Storage for workflows and executions
        self.workflows: Dict[str, WorkflowDefinition] = {}
        self.executions: Dict[str, WorkflowExecution] = {}
        self.running_executions: Dict[str, asyncio.Task] = {}
        
        # Execution limits
        self.max_total_executions = 10
        self.execution_lock = asyncio.Lock()
    
    def register_workflow(self, workflow: WorkflowDefinition) -> bool:
        """
        Register a new workflow definition
        
        Args:
            workflow: Workflow definition to register
            
        Returns:
            True if registration successful, False otherwise
        """
        try:
            # Validate workflow
            if not self._validate_workflow(workflow):
                return False
            
            # Check for circular dependencies
            if self._has_circular_dependencies(workflow):
                self.logger.error(f"Workflow {workflow.workflow_id} has circular dependencies")
                return False
            
            # Register workflow
            self.workflows[workflow.workflow_id] = workflow
            self.logger.info(f"Registered workflow: {workflow.name} ({workflow.workflow_id})")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register workflow {workflow.workflow_id}: {e}")
            self.error_handler.record_error(
                error=e,
                context={"workflow_id": workflow.workflow_id if workflow else "unknown"},
                component="WorkflowManager",
                operation="register_workflow"
            )
            return False
    
    def _validate_workflow(self, workflow: WorkflowDefinition) -> bool:
        """Validate workflow definition"""
        if not workflow.workflow_id or not workflow.name:
            return False
        
        if not workflow.steps:
            self.logger.error(f"Workflow {workflow.workflow_id} has no steps")
            return False
        
        # Validate step IDs are unique
        step_ids = [step.step_id for step in workflow.steps]
        if len(step_ids) != len(set(step_ids)):
            self.logger.error(f"Workflow {workflow.workflow_id} has duplicate step IDs")
            return False
        
        # Validate dependencies exist
        for step in workflow.steps:
            for dep in step.dependencies:
                if dep not in step_ids:
                    self.logger.error(f"Step {step.step_id} depends on non-existent step {dep}")
                    return False
        
        return True
    
    def _has_circular_dependencies(self, workflow: WorkflowDefinition) -> bool:
        """Check for circular dependencies in workflow steps"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(step_id: str) -> bool:
            if step_id in rec_stack:
                return True
            if step_id in visited:
                return False
            
            visited.add(step_id)
            rec_stack.add(step_id)
            
            # Find step by ID
            step = next((s for s in workflow.steps if s.step_id == step_id), None)
            if step:
                for dep in step.dependencies:
                    if has_cycle(dep):
                        return True
            
            rec_stack.remove(step_id)
            return False
        
        for step in workflow.steps:
            if step.step_id not in visited:
                if has_cycle(step.step_id):
                    return True
        
        return False
    
    @ErrorRecovery.retry(max_attempts=3, backoff_factor=2)
    async def execute_workflow(self, workflow_id: str, config: Optional[Dict[str, Any]] = None) -> str:
        """
        Execute a workflow
        
        Args:
            workflow_id: ID of the workflow to execute
            config: Optional configuration for this execution
            
        Returns:
            Execution ID
        """
        async with self.execution_lock:
            # Check if workflow exists
            if workflow_id not in self.workflows:
                raise ValueError(f"Workflow {workflow_id} not found")
            
            workflow = self.workflows[workflow_id]
            
            # Check execution limits
            if len(self.running_executions) >= self.max_total_executions:
                raise RuntimeError("Maximum number of concurrent executions reached")
            
            # Check workflow-specific limits
            running_count = sum(1 for exec_id, task in self.running_executions.items() 
                              if self.executions[exec_id].workflow_id == workflow_id)
            if running_count >= workflow.max_concurrent_executions:
                raise RuntimeError(f"Maximum concurrent executions for workflow {workflow_id} reached")
            
            # Create execution record
            execution_id = str(uuid.uuid4())
            execution = WorkflowExecution(
                execution_id=execution_id,
                workflow_id=workflow_id,
                status=WorkflowStatus.PENDING,
                start_time=datetime.now(),
                config=config or {}
            )
            
            # Initialize step statuses
            for step in workflow.steps:
                execution.steps[step.step_id] = StepStatus.PENDING
            
            # Store execution
            self.executions[execution_id] = execution
            
            # Start execution
            task = asyncio.create_task(self._execute_workflow_task(execution_id))
            self.running_executions[execution_id] = task
            
            self.logger.info(f"Started workflow execution: {execution_id} for workflow: {workflow_id}")
            
            return execution_id
    
    async def _execute_workflow_task(self, execution_id: str):
        """Internal task for workflow execution"""
        try:
            execution = self.executions[execution_id]
            workflow = self.workflows[execution.workflow_id]
            
            # Update status
            execution.status = WorkflowStatus.RUNNING
            
            # Execute steps in dependency order
            await self._execute_steps(execution, workflow)
            
            # Check if all required steps completed
            failed_steps = [step_id for step_id, status in execution.steps.items() 
                          if status == StepStatus.FAILED]
            
            if failed_steps:
                execution.status = WorkflowStatus.FAILED
                execution.errors.append(f"Failed steps: {', '.join(failed_steps)}")
            else:
                execution.status = WorkflowStatus.COMPLETED
            
        except Exception as e:
            execution.status = WorkflowStatus.FAILED
            execution.errors.append(f"Workflow execution failed: {str(e)}")
            self.logger.error(f"Workflow execution {execution_id} failed: {e}")
            self.error_handler.record_error(
                error=e,
                context={"execution_id": execution_id, "workflow_id": workflow_id},
                component="WorkflowManager",
                operation="execute_workflow"
            )
        
        finally:
            execution.end_time = datetime.now()
            
            # Clean up
            if execution_id in self.running_executions:
                del self.running_executions[execution_id]
    
    async def _execute_steps(self, execution: WorkflowExecution, workflow: WorkflowDefinition):
        """Execute workflow steps in dependency order"""
        # Build dependency graph
        step_dependencies = {step.step_id: set(step.dependencies) for step in workflow.steps}
        completed_steps = set()
        
        while len(completed_steps) < len(workflow.steps):
            # Find steps ready to execute
            ready_steps = []
            for step in workflow.steps:
                if (step.step_id not in completed_steps and 
                    step_dependencies[step.step_id].issubset(completed_steps)):
                    ready_steps.append(step)
            
            if not ready_steps:
                # Check for deadlock
                remaining_steps = [s.step_id for s in workflow.steps if s.step_id not in completed_steps]
                raise RuntimeError(f"Deadlock detected. Remaining steps: {remaining_steps}")
            
            # Execute ready steps in parallel
            tasks = []
            for step in ready_steps:
                task = asyncio.create_task(self._execute_step(execution, step))
                tasks.append(task)
            
            # Wait for all tasks to complete
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Process results
            for step, result in zip(ready_steps, results):
                if isinstance(result, Exception):
                    execution.steps[step.step_id] = StepStatus.FAILED
                    execution.errors.append(f"Step {step.step_id} failed: {result}")
                else:
                    execution.steps[step.step_id] = StepStatus.COMPLETED
                    execution.results[step.step_id] = result
                    completed_steps.add(step.step_id)
    
    async def _execute_step(self, execution: WorkflowExecution, step: WorkflowStep) -> Any:
        """Execute a single workflow step"""
        self.logger.info(f"Executing step: {step.name} ({step.step_id})")
        
        # Update status
        execution.steps[step.step_id] = StepStatus.RUNNING
        
        try:
            # Execute step with timeout and retry logic
            result = await asyncio.wait_for(
                self._execute_step_with_retry(step),
                timeout=step.timeout_seconds
            )
            
            self.logger.info(f"Step {step.step_id} completed successfully")
            return result
            
        except asyncio.TimeoutError:
            error_msg = f"Step {step.step_id} timed out after {step.timeout_seconds} seconds"
            self.logger.error(error_msg)
            execution.errors.append(error_msg)
            raise
            
        except Exception as e:
            error_msg = f"Step {step.step_id} failed: {str(e)}"
            self.logger.error(error_msg)
            execution.errors.append(error_msg)
            raise
    
    async def _execute_step_with_retry(self, step: WorkflowStep) -> Any:
        """Execute step with retry logic"""
        last_exception = None
        
        for attempt in range(step.retry_attempts):
            try:
                # Execute the step function
                if asyncio.iscoroutinefunction(step.function):
                    result = await step.function(**step.config)
                else:
                    result = step.function(**step.config)
                
                return result
                
            except Exception as e:
                last_exception = e
                if attempt < step.retry_attempts - 1:
                    self.logger.warning(f"Step {step.step_id} attempt {attempt + 1} failed, retrying in {step.retry_delay_seconds} seconds: {e}")
                    await asyncio.sleep(step.retry_delay_seconds)
        
        # All retries failed
        raise last_exception
    
    def get_execution_status(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution status by ID"""
        return self.executions.get(execution_id)
    
    def get_execution(self, execution_id: str) -> Optional[WorkflowExecution]:
        """Get execution by ID (alias for get_execution_status)"""
        return self.executions.get(execution_id)
    
    def list_executions(self, workflow_id: Optional[str] = None, 
                       status: Optional[WorkflowStatus] = None) -> List[WorkflowExecution]:
        """List executions with optional filtering"""
        executions = list(self.executions.values())
        
        if workflow_id:
            executions = [e for e in executions if e.workflow_id == workflow_id]
        
        if status:
            executions = [e for e in executions if e.status == status]
        
        return executions
    
    def cancel_execution(self, execution_id: str) -> bool:
        """Cancel a running execution"""
        if execution_id not in self.executions:
            return False
        
        execution = self.executions[execution_id]
        if execution.status != WorkflowStatus.RUNNING:
            return False
        
        # Cancel the task
        if execution_id in self.running_executions:
            self.running_executions[execution_id].cancel()
            del self.running_executions[execution_id]
        
        execution.status = WorkflowStatus.CANCELLED
        execution.end_time = datetime.now()
        
        self.logger.info(f"Cancelled execution: {execution_id}")
        return True
    
    def get_workflow_definition(self, workflow_id: str) -> Optional[WorkflowDefinition]:
        """Get workflow definition by ID"""
        return self.workflows.get(workflow_id)
    
    def list_workflows(self) -> List[WorkflowDefinition]:
        """List all registered workflows"""
        return list(self.workflows.values())
    
    def delete_workflow(self, workflow_id: str) -> bool:
        """Delete a workflow definition"""
        if workflow_id not in self.workflows:
            return False
        
        # Check if workflow is currently running
        running_count = sum(1 for exec_id, task in self.running_executions.items() 
                          if self.executions[execution_id].workflow_id == workflow_id)
        
        if running_count > 0:
            self.logger.warning(f"Cannot delete workflow {workflow_id}: {running_count} executions running")
            return False
        
        del self.workflows[workflow_id]
        self.logger.info(f"Deleted workflow: {workflow_id}")
        return True
