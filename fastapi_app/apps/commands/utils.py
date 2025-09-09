from sqlalchemy.orm import Session
from sqlalchemy import desc
from typing import List, Optional
import subprocess
import uuid
import asyncio
from datetime import datetime
import json

from .models import CommandExecution
from .schemas import CommandRequest, CommandResponse, CommandHistory, CommandStatus

class CommandService:
    def __init__(self, db: Session):
        self.db = db
    
    async def execute_command(self, request: CommandRequest, background_tasks) -> CommandResponse:
        """Execute a command and return the result"""
        command_id = str(uuid.uuid4())
        start_time = datetime.now()
        
        # Create command execution record
        command_execution = CommandExecution(
            command_id=command_id,
            command=request.command,
            status=CommandStatus.PENDING,
            start_time=start_time
        )
        
        self.db.add(command_execution)
        self.db.commit()
        
        if request.background:
            # Run in background
            background_tasks.add_task(self._execute_background_command, command_id, request.command)
            return CommandResponse(
                command_id=command_id,
                command=request.command,
                status=CommandStatus.PENDING,
                start_time=start_time
            )
        else:
            # Run synchronously
            return await self._execute_sync_command(command_id, request.command)
    
    async def _execute_sync_command(self, command_id: str, command: str) -> CommandResponse:
        """Execute command synchronously"""
        start_time = datetime.now()
        
        try:
            # Update status to running
            command_execution = self.db.query(CommandExecution).filter(
                CommandExecution.command_id == command_id
            ).first()
            
            if command_execution:
                command_execution.status = CommandStatus.RUNNING
                self.db.commit()
            
            # Execute the command
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update command execution record
            if command_execution:
                command_execution.status = CommandStatus.COMPLETED if result.returncode == 0 else CommandStatus.FAILED
                command_execution.output = result.stdout
                command_execution.error = result.stderr
                command_execution.end_time = end_time
                command_execution.duration = duration
                self.db.commit()
            
            return CommandResponse(
                command_id=command_id,
                command=command,
                status=CommandStatus.COMPLETED if result.returncode == 0 else CommandStatus.FAILED,
                output=result.stdout,
                error=result.stderr,
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
            
        except subprocess.TimeoutExpired:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update command execution record
            command_execution = self.db.query(CommandExecution).filter(
                CommandExecution.command_id == command_id
            ).first()
            
            if command_execution:
                command_execution.status = CommandStatus.FAILED
                command_execution.error = "Command timed out after 5 minutes"
                command_execution.end_time = end_time
                command_execution.duration = duration
                self.db.commit()
            
            return CommandResponse(
                command_id=command_id,
                command=command,
                status=CommandStatus.FAILED,
                error="Command timed out after 5 minutes",
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
            
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update command execution record
            command_execution = self.db.query(CommandExecution).filter(
                CommandExecution.command_id == command_id
            ).first()
            
            if command_execution:
                command_execution.status = CommandStatus.FAILED
                command_execution.error = str(e)
                command_execution.end_time = end_time
                command_execution.duration = duration
                self.db.commit()
            
            return CommandResponse(
                command_id=command_id,
                command=command,
                status=CommandStatus.FAILED,
                error=str(e),
                start_time=start_time,
                end_time=end_time,
                duration=duration
            )
    
    async def _execute_background_command(self, command_id: str, command: str):
        """Execute command in background"""
        start_time = datetime.now()
        
        try:
            # Update status to running
            command_execution = self.db.query(CommandExecution).filter(
                CommandExecution.command_id == command_id
            ).first()
            
            if command_execution:
                command_execution.status = CommandStatus.RUNNING
                self.db.commit()
            
            # Execute the command
            result = subprocess.run(
                command.split(),
                capture_output=True,
                text=True,
                timeout=1800  # 30 minute timeout for background tasks
            )
            
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update command execution record
            if command_execution:
                command_execution.status = CommandStatus.COMPLETED if result.returncode == 0 else CommandStatus.FAILED
                command_execution.output = result.stdout
                command_execution.error = result.stderr
                command_execution.end_time = end_time
                command_execution.duration = duration
                self.db.commit()
                
        except Exception as e:
            end_time = datetime.now()
            duration = (end_time - start_time).total_seconds()
            
            # Update command execution record
            command_execution = self.db.query(CommandExecution).filter(
                CommandExecution.command_id == command_id
            ).first()
            
            if command_execution:
                command_execution.status = CommandStatus.FAILED
                command_execution.error = str(e)
                command_execution.end_time = end_time
                command_execution.duration = duration
                self.db.commit()
    
    def get_command_history(self, limit: int = 50) -> List[CommandHistory]:
        """Get command execution history"""
        command_executions = self.db.query(CommandExecution).order_by(
            desc(CommandExecution.start_time)
        ).limit(limit).all()
        
        return [
            CommandHistory(
                command_id=cmd.command_id,
                command=cmd.command,
                status=cmd.status,
                output=cmd.output,
                error=cmd.error,
                start_time=cmd.start_time,
                end_time=cmd.end_time,
                duration=cmd.duration
            )
            for cmd in command_executions
        ]

