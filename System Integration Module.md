```python
# system_integration.py

import asyncio
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import logging
import json
from typing import Dict, List, Optional, Any, Union, Callable, Set
from pathlib import Path
import aiohttp
import consul.aio
from aiokafka import AIOKafkaProducer, AIOKafkaConsumer
import etcd3.aio as etcd
import yaml
import socket
import uuid
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)

class ServiceStatus(Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    DEGRADED = "degraded"
    UNKNOWN = "unknown"

@dataclass
class ServiceConfig:
    """Configuration for a service."""
    name: str
    host: str
    port: int
    health_check_endpoint: str = "/health"
    health_check_interval: int = 30
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MessageConfig:
    """Configuration for message queues."""
    topic: str
    partition: int = 0
    retention_ms: int = 604800000  # 7 days
    replication_factor: int = 1

@dataclass
class EventConfig:
    """Configuration for events."""
    name: str
    handlers: List[str]
    retry_count: int = 3
    timeout: int = 30

class ServiceDiscovery:
    """Manages service discovery using Consul."""
    
    def __init__(self, host: str, port: int):
        self.client = consul.aio.Consul(host=host, port=port)
        self.services: Dict[str, ServiceConfig] = {}
        
    async def register_service(self, config: ServiceConfig):
        """Register a service with Consul."""
        service_id = f"{config.name}-{uuid.uuid4()}"
        
        try:
            await self.client.agent.service.register(
                name=config.name,
                service_id=service_id,
                address=config.host,
                port=config.port,
                tags=config.tags,
                check={
                    "http": f"http://{config.host}:{config.port}{config.health_check_endpoint}",
                    "interval": f"{config.health_check_interval}s"
                }
            )
            
            self.services[config.name] = config
            logger.info(f"Registered service: {config.name}")
            
        except Exception as e:
            logger.error(f"Failed to register service {config.name}: {e}")
            raise

    async def deregister_service(self, service_name: str):
        """Deregister a service from Consul."""
        try:
            if service_name in self.services:
                await self.client.agent.service.deregister(service_name)
                del self.services[service_name]
                logger.info(f"Deregistered service: {service_name}")
                
        except Exception as e:
            logger.error(f"Failed to deregister service {service_name}: {e}")
            raise

    async def discover_service(self, service_name: str) -> Optional[ServiceConfig]:
        """Discover a service by name."""
        try:
            _, services = await self.client.health.service(service_name, passing=True)
            
            if services:
                service = services[0]
                return ServiceConfig(
                    name=service["Service"]["Service"],
                    host=service["Service"]["Address"],
                    port=service["Service"]["Port"],
                    tags=service["Service"]["Tags"]
                )
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to discover service {service_name}: {e}")
            raise

class MessageBroker:
    """Manages message queuing using Kafka."""
    
    def __init__(self, bootstrap_servers: List[str]):
        self.bootstrap_servers = bootstrap_servers
        self.producer: Optional[AIOKafkaProducer] = None
        self.consumers: Dict[str, AIOKafkaConsumer] = {}
        
    async def start(self):
        """Start the message broker."""
        self.producer = AIOKafkaProducer(
            bootstrap_servers=self.bootstrap_servers
        )
        await self.producer.start()
        
    async def stop(self):
        """Stop the message broker."""
        if self.producer:
            await self.producer.stop()
            
        for consumer in self.consumers.values():
            await consumer.stop()

    async def publish_message(self, topic: str, message: Dict[str, Any]):
        """Publish a message to a topic."""
        if not self.producer:
            raise RuntimeError("Producer not started")
            
        try:
            value = json.dumps(message).encode()
            await self.producer.send_and_wait(topic, value)
            
        except Exception as e:
            logger.error(f"Failed to publish message to {topic}: {e}")
            raise

    async def subscribe(self, config: MessageConfig, handler: Callable):
        """Subscribe to a topic."""
        try:
            consumer = AIOKafkaConsumer(
                config.topic,
                bootstrap_servers=self.bootstrap_servers,
                group_id=f"{config.topic}-group"
            )
            
            await consumer.start()
            self.consumers[config.topic] = consumer
            
            async def consume():
                try:
                    async for msg in consumer:
                        try:
                            value = json.loads(msg.value.decode())
                            await handler(value)
                        except Exception as e:
                            logger.error(f"Error processing message: {e}")
                except Exception as e:
                    logger.error(f"Consumer error: {e}")
                    
            asyncio.create_task(consume())
            
        except Exception as e:
            logger.error(f"Failed to subscribe to {config.topic}: {e}")
            raise

class EventBus:
    """Manages event distribution using etcd."""
    
    def __init__(self, host: str, port: int):
        self.client = etcd.client(host=host, port=port)
        self.handlers: Dict[str, List[Callable]] = {}
        self.watching = False
        
    async def start(self):
        """Start the event bus."""
        self.watching = True
        asyncio.create_task(self._watch_events())
        
    async def stop(self):
        """Stop the event bus."""
        self.watching = False
        await self.client.close()

    async def publish_event(self, event: EventConfig, data: Dict[str, Any]):
        """Publish an event."""
        try:
            event_data = {
                "name": event.name,
                "timestamp": datetime.now().isoformat(),
                "data": data
            }
            
            await self.client.put(
                f"/events/{event.name}",
                json.dumps(event_data)
            )
            
        except Exception as e:
            logger.error(f"Failed to publish event {event.name}: {e}")
            raise

    async def subscribe_to_event(self, event_name: str, handler: Callable):
        """Subscribe to an event."""
        if event_name not in self.handlers:
            self.handlers[event_name] = []
        self.handlers[event_name].append(handler)

    async def _watch_events(self):
        """Watch for events in etcd."""
        while self.watching:
            try:
                async for event in self.client.watch_prefix("/events/"):
                    if event.type == "PUT":
                        event_data = json.loads(event.value)
                        event_name = event_data["name"]
                        
                        if event_name in self.handlers:
                            for handler in self.handlers[event_name]:
                                try:
                                    await handler(event_data["data"])
                                except Exception as e:
                                    logger.error(f"Error in event handler: {e}")
                                    
            except Exception as e:
                logger.error(f"Error watching events: {e}")
                await asyncio.sleep(5)

class ConfigManager:
    """Manages configuration using etcd."""
    
    def __init__(self, host: str, port: int):
        self.client = etcd3.client(host=host, port=port)  # Changed to sync client
        self.cache: Dict[str, Any] = {}
        
    async def set_config(self, key: str, value: Any):
        """Set a configuration value."""
        try:
            # Convert to sync operation
            self.client.put(
                f"/config/{key}",
                json.dumps(value).encode()
            )
            self.cache[key] = value
            
        except Exception as e:
            logger.error(f"Failed to set config {key}: {e}")
            raise

    async def get_config(self, key: str) -> Optional[Any]:
        """Get a configuration value."""
        try:
            if key in self.cache:
                return self.cache[key]
                
            # Convert to sync operation
            result = self.client.get(f"/config/{key}")
            if result[0]:
                value = json.loads(result[0].decode())
                self.cache[key] = value
                return value
                
            return None
            
        except Exception as e:
            logger.error(f"Failed to get config {key}: {e}")
            raise

class HealthMonitor:
    """Monitors service health."""
    
    def __init__(self, service_discovery: ServiceDiscovery):
        self.service_discovery = service_discovery
        self.health_checks: Dict[str, asyncio.Task] = {}
        
    async def start_monitoring(self, service: ServiceConfig):
        """Start monitoring a service."""
        if service.name not in self.health_checks:
            self.health_checks[service.name] = asyncio.create_task(
                self._check_service_health(service))
            
    async def stop_monitoring(self, service_name: str):
        """Stop monitoring a service."""
        if service_name in self.health_checks:
            self.health_checks[service_name].cancel()
            del self.health_checks[service_name]

    async def _check_service_health(self, service: ServiceConfig):
        """Periodically check service health."""
        async with aiohttp.ClientSession() as session:
            while True:
                try:
                    async with session.get(
                        f"http://{service.host}:{service.port}{service.health_check_endpoint}"
                    ) as response:
                        if response.status == 200:
                            status = ServiceStatus.HEALTHY
                        else:
                            status = ServiceStatus.DEGRADED
                except Exception:
                    status = ServiceStatus.UNHEALTHY
                    
                # Update service status in Consul
                await self.service_discovery.client.agent.service.maintenance(
                    service.name,
                    enable=status != ServiceStatus.HEALTHY,
                    reason=f"Health check status: {status.value}"
                )
                
                await asyncio.sleep(service.health_check_interval)

class SystemIntegrator:
    """Main class for system integration."""
    
    def __init__(self,
                 consul_host: str,
                 consul_port: int,
                 kafka_servers: List[str],
                 etcd_host: str,
                 etcd_port: int):
        self.service_discovery = ServiceDiscovery(consul_host, consul_port)
        self.message_broker = MessageBroker(kafka_servers)
        self.event_bus = EventBus(etcd_host, etcd_port)
        self.config_manager = ConfigManager(etcd_host, etcd_port)
        self.health_monitor = HealthMonitor(self.service_discovery)
        
    async def start(self):
        """Start all integration components."""
        await self.message_broker.start()
        await self.event_bus.start()
        
    async def stop(self):
        """Stop all integration components."""
        await self.message_broker.stop()
        await self.event_bus.stop()

    async def register_service(self, config: ServiceConfig):
        """Register a service and start health monitoring."""
        await self.service_discovery.register_service(config)
        await self.health_monitor.start_monitoring(config)

    async def deregister_service(self, service_name: str):
        """Deregister a service and stop health monitoring."""
        await self.service_discovery.deregister_service(service_name)
        await self.health_monitor.stop_monitoring(service_name)

# CLI support
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="System Integration")
    parser.add_argument("--consul-host", default="localhost",
                       help="Consul host")
    parser.add_argument("--consul-port", type=int, default=8500,
                       help="Consul port")
    parser.add_argument("--kafka-servers", nargs="+", required=True,
                       help="Kafka bootstrap servers")
    parser.add_argument("--etcd-host", default="localhost",
                       help="etcd host")
    parser.add_argument("--etcd-port", type=int, default=2379,
                       help="etcd port")
    
    args = parser.parse_args()
    
    async def main():
        integrator = SystemIntegrator(
            args.consul_host,
            args.consul_port,
            args.kafka_servers,
            args.etcd_host,
            args.etcd_port
        )
        
        try:
            await integrator.start()
            
            # Keep the system running
            while True:
                await asyncio.sleep(1)
                
        except KeyboardInterrupt:
            await integrator.stop()
    
    asyncio.run(main())

```