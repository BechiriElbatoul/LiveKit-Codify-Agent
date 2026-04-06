from livekit.agents.beta.workflow import TaskGroup
from livekit.agents import Agent, AgentTask, function_tool, RunContext, get_job_context
from dataclasses import dataclass


@dataclass
class EmailResult:
    email_address: str  


@dataclass
class AddressResult:
    address: str 


class GetEmailTask(AgentTask[EmailResult]): 
    def __init__(self) -> None:
        super().__init__(
            instructions="Collect the user's email address."
        )
    
    @function_tool
    async def record_email(self, context: RunContext, email: str) -> None: 
        """Record the user's email address"""
        self.complete(EmailResult(email_address=email))


class GetAddressTask(AgentTask[AddressResult]): 
    def __init__(self) -> None:
        super().__init__(
            instructions="Collect the user's shipping address."  
        )
    
    @function_tool
    async def record_address(self, context: RunContext, address: str) -> None:
        """Record the user's shipping address"""
        self.complete(AddressResult(address=address))


class CheckoutAgent(Agent):
    async def on_enter(self) -> None:
        task_group = TaskGroup()
        task_group.add(
            lambda: GetEmailTask(),
            id="email", 
            description="Collect email address"
        )
        task_group.add(
            lambda: GetAddressTask(),
            id="address",  
            description="Collect shipping address"
        )
        
        results = await task_group 
        
        email = results.task_results["email"].email_address
        address = results.task_results["address"].address
        
        await self.session.generate_reply(
            instructions=f"Confirm the order will be sent to {email} at {address}."  
        )
    
    @function_tool
    async def transfer_to_human(self, context: RunContext) -> None:
        """Transfer the call to a human agent."""
        context.disallow_interruptions()
        
        room = context.session.room
        
        await room.local_participant.publish_sip_participant(
            sip_trunk_id="trunk id ?", 
            dial_to="sip uri ? using twilio or so", 
        )