```python
# core/storage/__init__.py
"""
Storage Package.

This package provides integrated storage functionality with support for database operations,
metadata management, and caching. All components implement lifecycle management and
event system integration.
"""

from core.storage.database import Database, DatabaseError
from core.storage.metadata import MetadataStorage, MetadataError
from core.storage.cache import Cache, CacheError

__all__ = [
    'Database',
    'DatabaseError',
    'MetadataStorage',
    'MetadataError',
    'Cache',
    'CacheError'
]
```


```python
# core/storage/database.py
"""Database management with lifecycle and event integration."""

import asyncio
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any

import aiosqlite

from core.events import EventBus, Event, EventType
from core.lifecycle import LifecycleComponent, ComponentState

class DatabaseError(Exception):
    """Base exception for database-related errors."""
    pass

class Database(LifecycleComponent):
    """Integrated database management with lifecycle support."""
    
    def __init__(self, name: str, event_bus: EventBus, path: str, max_connections: int = 5):
        """Initialize database component."""
        super().__init__(name, event_bus)
        self.db_path = Path(path)
        self.max_connections = max_connections
        self._pool_semaphore = asyncio.Semaphore(max_connections)
        self._connections: List[aiosqlite.Connection] = []
        self._initialized = False

    async def _do_initialize(self) -> None:
        """Initialize database and create tables."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            
            async with self._get_connection() as conn:
                await self._create_tables(conn)
            
            self._initialized = True
            
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_INITIALIZED,
                data={
                    'component': self.name,
                    'database': str(self.db_path),
                    'tables_created': True
                }
            ))
            
        except Exception as e:
            raise DatabaseError(f"Database initialization failed: {str(e)}") from e

    async def _do_start(self) -> None:
        """Start database operations."""
        # Pre-initialize connection pool
        for _ in range(self.max_connections):
            conn = await self._create_connection()
            self._connections.append(conn)

    async def _do_stop(self) -> None:
        """Stop database operations."""
        # Close all connections
        for conn in self._connections:
            await conn.close()
        self._connections.clear()

    async def _do_cleanup(self) -> None:
        """Clean up database resources."""
        if self._initialized:
            await self.vacuum()

    @asyncio.contextmanager
    async def _get_connection(self):
        """Get a database connection from the pool."""
        async with self._pool_semaphore:
            # Get or create connection
            if self._connections:
                conn = self._connections.pop()
            else:
                conn = await self._create_connection()
            
            try:
                yield conn
            except Exception as e:
                await self.event_bus.publish(Event(
                    type=EventType.COMPONENT_ERROR,
                    data={
                        'component': self.name,
                        'error': str(e),
                        'operation': 'database_operation'
                    }
                ))
                raise
            finally:
                self._connections.append(conn)

    async def _create_connection(self) -> aiosqlite.Connection:
        """Create a new database connection."""
        conn = await aiosqlite.connect(
            self.db_path,
            isolation_level=None  # Enable autocommit mode
        )
        await conn.execute("PRAGMA journal_mode=WAL")
        return conn

    async def execute(self, query: str, params: Optional[tuple] = None) -> None:
        """Execute a query with error handling and events."""
        if not self._initialized:
            raise DatabaseError("Database not initialized")

        try:
            async with self._get_connection() as conn:
                await conn.execute(query, params or ())
                await conn.commit()
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'execute_query'
                }
            ))
            raise DatabaseError(f"Query execution failed: {str(e)}") from e

    async def fetch_one(self, query: str, params: Optional[tuple] = None) -> Optional[Dict[str, Any]]:
        """Fetch a single row with error handling."""
        if not self._initialized:
            raise DatabaseError("Database not initialized")

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(query, params or ())
                row = await cursor.fetchone()
                return dict(row) if row else None
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'fetch_one'
                }
            ))
            raise DatabaseError(f"Query failed: {str(e)}") from e

    async def fetch_all(self, query: str, params: Optional[tuple] = None) -> List[Dict[str, Any]]:
        """Fetch all rows with error handling."""
        if not self._initialized:
            raise DatabaseError("Database not initialized")

        try:
            async with self._get_connection() as conn:
                cursor = await conn.execute(query, params or ())
                rows = await cursor.fetchall()
                return [dict(row) for row in rows]
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'fetch_all'
                }
            ))
            raise DatabaseError(f"Query failed: {str(e)}") from e

    async def vacuum(self) -> None:
        """Optimize database storage."""
        try:
            async with self._get_connection() as conn:
                await conn.execute("VACUUM")
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'vacuum'
                }
            ))
            raise DatabaseError(f"Vacuum failed: {str(e)}") from e

    async def get_stats(self) -> Dict[str, Any]:
        """Get database statistics."""
        try:
            stats = {
                'path': str(self.db_path),
                'size_bytes': self.db_path.stat().st_size if self.db_path.exists() else 0,
                'connections': len(self._connections),
                'max_connections': self.max_connections
            }
            
            async with self._get_connection() as conn:
                # Get page stats
                cursor = await conn.execute("PRAGMA page_count")
                stats['page_count'] = (await cursor.fetchone())[0]
                
                cursor = await conn.execute("PRAGMA page_size")
                stats['page_size'] = (await cursor.fetchone())[0]
                
                # Get table stats
                cursor = await conn.execute("""
                    SELECT name FROM sqlite_master 
                    WHERE type='table'
                """)
                tables = await cursor.fetchall()
                
                stats['tables'] = {}
                for (table_name,) in tables:
                    cursor = await conn.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = (await cursor.fetchone())[0]
                    stats['tables'][table_name] = {'row_count': row_count}
            
            return stats
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'get_stats'
                }
            ))
            raise DatabaseError(f"Failed to get stats: {str(e)}") from e
```

```python
# core/storage/metadata.py
"""Metadata storage with lifecycle and event integration."""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional, Any

from core.events import EventBus, Event, EventType
from core.lifecycle import LifecycleComponent
from core.storage.database import Database, DatabaseError


class MetadataError(Exception):
    """Base exception for metadata-related errors."""
    pass


class MetadataStorage(LifecycleComponent):
    """Integrated metadata storage with lifecycle support."""
    
    def __init__(self, name: str, event_bus: EventBus, database: Database):
        """Initialize metadata storage."""
        super().__init__(name, event_bus)
        self.database = database
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._cache_lock = asyncio.Lock()

    async def _do_initialize(self) -> None:
        """Initialize metadata storage."""
        try:
            # Create metadata table
            await self.database.execute("""
                CREATE TABLE IF NOT EXISTS metadata (
                    key TEXT PRIMARY KEY,
                    data JSON NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            await self.database.execute("""
                CREATE INDEX IF NOT EXISTS idx_metadata_updated 
                ON metadata(updated_at)
            """)
            
        except DatabaseError as e:
            raise MetadataError(f"Failed to initialize metadata storage: {str(e)}")

    async def _do_start(self) -> None:
        """Start metadata operations."""
        # Warm up cache with recent items
        try:
            recent_items = await self.database.fetch_all(
                "SELECT key, data FROM metadata ORDER BY updated_at DESC LIMIT 100"
            )
            
            async with self._cache_lock:
                for item in recent_items:
                    self._cache[item['key']] = json.loads(item['data'])
                    
        except DatabaseError as e:
            self._logger.warning(f"Cache warm-up failed: {str(e)}")

    async def _do_stop(self) -> None:
        """Stop metadata operations."""
        async with self._cache_lock:
            self._cache.clear()

    async def _do_cleanup(self) -> None:
        """Clean up metadata resources."""
        # No specific cleanup needed
        pass

    async def store(self, key: str, metadata: Dict[str, Any]) -> None:
        """Store metadata with event notification."""
        try:
            # Store in database
            await self.database.execute(
                """
                INSERT INTO metadata (key, data, updated_at)
                VALUES (?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(key) DO UPDATE SET
                    data = ?,
                    updated_at = CURRENT_TIMESTAMP
                """,
                (key, json.dumps(metadata), json.dumps(metadata))
            )
            
            # Update cache
            async with self._cache_lock:
                self._cache[key] = metadata.copy()
            
            # Publish event
            await self.event_bus.publish(Event(
                type=EventType.CACHE_UPDATED,
                data={
                    'key': key,
                    'operation': 'store'
                }
            ))
            
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'store'
                }
            ))
            raise MetadataError(f"Failed to store metadata: {str(e)}")

    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """Get metadata with cache."""
        try:
            # Check cache first
            async with self._cache_lock:
                if key in self._cache:
                    return self._cache[key].copy()
            
            # Get from database
            result = await self.database.fetch_one(
                "SELECT data FROM metadata WHERE key = ?",
                (key,)
            )
            
            if result:
                data = json.loads(result['data'])
                # Update cache
                async with self._cache_lock:
                    self._cache[key] = data.copy()
                return data
                
            return None
            
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'get'
                }
            ))
            raise MetadataError(f"Failed to get metadata: {str(e)}")

    async def delete(self, key: str) -> bool:
        """Delete metadata with event notification."""
        try:
            # Remove from database
            await self.database.execute(
                "DELETE FROM metadata WHERE key = ?",
                (key,)
            )
            
            # Remove from cache
            async with self._cache_lock:
                was_cached = key in self._cache
                self._cache.pop(key, None)
            
            # Publish event
            if was_cached:
                await self.event_bus.publish(Event(
                    type=EventType.CACHE_UPDATED,
                    data={
                        'key': key,
                        'operation': 'delete'
                    }
                ))
            
            return was_cached
            
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'delete'
                }
            ))
            raise MetadataError(f"Failed to delete metadata: {str(e)}")

    async def get_stats(self) -> Dict[str, Any]:
        """Get metadata storage statistics."""
        try:
            async with self._cache_lock:
                cache_size = len(self._cache)
                cache_keys = list(self._cache.keys())

            stats = {
                'total_entries': await self.database.fetch_one(
                    "SELECT COUNT(*) as count FROM metadata"
                ),
                'cache_entries': cache_size,
                'cache_keys': cache_keys[:100],  # First 100 keys
                'recent_updates': await self.database.fetch_all(
                    """
                    SELECT key, updated_at 
                    FROM metadata 
                    ORDER BY updated_at DESC 
                    LIMIT 10
                    """
                ),
                'storage_size': await self.database.fetch_one(
                    "SELECT SUM(LENGTH(data)) as total_size FROM metadata"
                )
            }
            
            return stats
            
        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'get_stats'
                }
            ))
            raise MetadataError(f"Failed to get stats: {str(e)}")
```

```python
# core/storage/cache.py
"""Cache system with lifecycle and event integration."""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple

from core.events import EventBus, Event, EventType
from core.lifecycle import LifecycleComponent

class CacheError(Exception):
    """Base exception for cache-related errors."""
    pass

@dataclass
class CacheEntry:
    """Represents a cached item with metadata."""
    data: Any
    expiry: datetime
    size: int
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0

class Cache(LifecycleComponent):
    """Integrated cache system with lifecycle support."""
    
    def __init__(
        self,
        name: str,
        event_bus: EventBus,
        max_size: int = 1024 * 1024 * 100,  # 100MB default
        default_ttl: timedelta = timedelta(hours=24)
    ):
        """Initialize the cache component."""
        super().__init__(name, event_bus)
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._cache: Dict[str, CacheEntry] = {}
        self._current_size = 0
        self._lock = asyncio.Lock()
        
        # Start cleanup task when running
        self._cleanup_task: Optional[asyncio.Task] = None

    async def _do_initialize(self) -> None:
        """Initialize cache system."""
        pass  # No initialization needed

    async def _do_start(self) -> None:
        """Start cache operations and cleanup task."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def _do_stop(self) -> None:
        """Stop cache operations and cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None

    async def _do_cleanup(self) -> None:
        """Clean up cache resources."""
        async with self._lock:
            self._cache.clear()
            self._current_size = 0

    async def get(self, key: str) -> Optional[Any]:
        """Get item from cache with metrics."""
        async with self._lock:
            entry = self._cache.get(key)
            if entry:
                if entry.expiry > datetime.now():
                    entry.last_accessed = datetime.now()
                    entry.access_count += 1
                    return entry.data
                else:
                    await self._remove_entry(key)
        return None

    async def set(
        self,
        key: str,
        data: Any,
        ttl: Optional[timedelta] = None
    ) -> None:
        """Set cache item with event notification."""
        expiry = datetime.now() + (ttl or self.default_ttl)
        size = self._estimate_size(data)

        async with self._lock:
            # Remove old entry if exists
            if key in self._cache:
                await self._remove_entry(key)

            # Enforce size limit
            while self._current_size + size > self.max_size:
                await self._evict_lru()

            # Add new entry
            entry = CacheEntry(
                data=data,
                expiry=expiry,
                size=size
            )
            self._cache[key] = entry
            self._current_size += size

            # Publish event
            await self.event_bus.publish(Event(
                type=EventType.CACHE_UPDATED,
                data={
                    'key': key,
                    'operation': 'set',
                    'size': size
                }
            ))

    async def delete(self, key: str) -> bool:
        """Delete cache item with event notification."""
        async with self._lock:
            if key in self._cache:
                await self._remove_entry(key)
                
                # Publish event
                await self.event_bus.publish(Event(
                    type=EventType.CACHE_UPDATED,
                    data={
                        'key': key,
                        'operation': 'delete'
                    }
                ))
                return True
        return False

    async def _remove_entry(self, key: str) -> None:
        """Remove a cache entry and update size."""
        entry = self._cache.pop(key, None)
        if entry:
            self._current_size -= entry.size

    async def _evict_lru(self) -> None:
        """Evict the least recently used entry."""
        if not self._cache:
            return

        # Find entry with oldest last_accessed
        lru_key = min(
            self._cache.items(),
            key=lambda x: x[1].last_accessed
        )[0]
        await self._remove_entry(lru_key)

        await self.event_bus.publish(Event(
            type=EventType.CACHE_UPDATED,
            data={
                'key': lru_key,
                'operation': 'evict'
            }
        ))

    async def _cleanup_loop(self) -> None:
        """Background task to clean up expired entries."""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_expired()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cleanup error: {str(e)}")

    async def _cleanup_expired(self) -> None:
        """Clean up expired cache entries."""
        async with self._lock:
            now = datetime.now()
            expired_keys = [
                key for key, entry in self._cache.items()
                if entry.expiry <= now
            ]
            
            for key in expired_keys:
                await self._remove_entry(key)
                
                await self.event_bus.publish(Event(
                    type=EventType.CACHE_UPDATED,
                    data={
                        'key': key,
                        'operation': 'expire'
                    }
                ))

    def _estimate_size(self, data: Any) -> int:
        """Estimate size of data in bytes."""
        import sys
        return sys.getsizeof(data)

    async def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        async with self._lock:
            stats = {
                'current_size': self._current_size,
                'max_size': self.max_size,
                'utilization': self._current_size / self.max_size,
                'num_entries': len(self._cache),
                'entry_stats': {
                    key: {
                        'size': entry.size,
                        'access_count': entry.access_count,
                        'age_seconds': (datetime.now() - entry.expiry).total_seconds(),
                        'last_accessed': entry.last_accessed.isoformat()
                    }
                    for key, entry in self._cache.items()
                }
            }
            return stats
```


```python
"""
Component Lifecycle Management Module.

This module provides base classes and utilities for managing component lifecycles,
including initialization, startup, shutdown, and cleanup procedures.
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum, auto
from typing import Dict, List, Optional, Any

from core.events import EventBus, Event, EventType


class ComponentState(Enum):
    """Component lifecycle states."""
    CREATED = auto()
    INITIALIZING = auto()
    INITIALIZED = auto()
    STARTING = auto()
    RUNNING = auto()
    STOPPING = auto()
    STOPPED = auto()
    ERROR = auto()


@dataclass
class ComponentStatus:
    """Component status information."""
    state: ComponentState
    last_updated: datetime
    error: Optional[str] = None
    details: Dict[str, Any] = None


class LifecycleComponent(ABC):
    """
    Base class for components with lifecycle management.
    
    This class provides the basic structure for component lifecycle management
    including initialization, startup, shutdown, and cleanup procedures.
    """
    
    def __init__(self, name: str, event_bus: EventBus):
        """
        Initialize the component.
        
        Args:
            name: Component name
            event_bus: Event bus for lifecycle events
        """
        self.name = name
        self.event_bus = event_bus
        self._state = ComponentState.CREATED
        self._status = ComponentStatus(
            state=ComponentState.CREATED,
            last_updated=datetime.now()
        )
        self._lock = asyncio.Lock()
        self._logger = logging.getLogger(name)

    @property
    def state(self) -> ComponentState:
        """Get current component state."""
        return self._state

    async def initialize(self) -> None:
        """Initialize the component."""
        async with self._lock:
            if self._state != ComponentState.CREATED:
                raise RuntimeError(f"Cannot initialize component in state: {self._state}")
            
            try:
                self._update_state(ComponentState.INITIALIZING)
                await self._do_initialize()
                self._update_state(ComponentState.INITIALIZED)
                
                await self.event_bus.publish(Event(
                    type=EventType.COMPONENT_INITIALIZED,
                    data={
                        'component': self.name,
                        'state': self._state.name
                    }
                ))
                
            except Exception as e:
                self._handle_error("Initialization failed", e)
                raise

    async def start(self) -> None:
        """Start the component."""
        async with self._lock:
            if self._state != ComponentState.INITIALIZED:
                raise RuntimeError(f"Cannot start component in state: {self._state}")
            
            try:
                self._update_state(ComponentState.STARTING)
                await self._do_start()
                self._update_state(ComponentState.RUNNING)
                
                await self.event_bus.publish(Event(
                    type=EventType.COMPONENT_STARTED,
                    data={
                        'component': self.name,
                        'state': self._state.name
                    }
                ))
                
            except Exception as e:
                self._handle_error("Start failed", e)
                raise

    async def stop(self) -> None:
        """Stop the component."""
        async with self._lock:
            if self._state != ComponentState.RUNNING:
                raise RuntimeError(f"Cannot stop component in state: {self._state}")
            
            try:
                self._update_state(ComponentState.STOPPING)
                await self._do_stop()
                self._update_state(ComponentState.STOPPED)
                
                await self.event_bus.publish(Event(
                    type=EventType.COMPONENT_STOPPED,
                    data={
                        'component': self.name,
                        'state': self._state.name
                    }
                ))
                
            except Exception as e:
                self._handle_error("Stop failed", e)
                raise

    async def cleanup(self) -> None:
        """Clean up component resources."""
        try:
            await self._do_cleanup()
        except Exception as e:
            self._logger.error(f"Cleanup failed: {str(e)}")
            raise

    @abstractmethod
    async def _do_initialize(self) -> None:
        """Perform component initialization."""
        pass

    @abstractmethod
    async def _do_start(self) -> None:
        """Perform component startup."""
        pass

    @abstractmethod
    async def _do_stop(self) -> None:
        """Perform component shutdown."""
        pass

    @abstractmethod
    async def _do_cleanup(self) -> None:
        """Perform component cleanup."""
        pass

    def _update_state(self, state: ComponentState) -> None:
        """Update component state."""
        self._state = state
        self._status = ComponentStatus(
            state=state,
            last_updated=datetime.now()
        )
        self._logger.info(f"Component {self.name} state changed to {state.name}")

    def _handle_error(self, message: str, error: Exception) -> None:
        """Handle component error."""
        self._state = ComponentState.ERROR
        self._status = ComponentStatus(
            state=ComponentState.ERROR,
            last_updated=datetime.now(),
            error=str(error)
        )
        self._logger.error(f"{message}: {str(error)}")


class ApplicationLifecycle:
    """
    Manages lifecycle of multiple components.
    
    This class coordinates initialization, startup, and shutdown
    of multiple components while maintaining proper order and
    handling dependencies.
    """
    
    def __init__(self, event_bus: EventBus):
        """
        Initialize the application lifecycle manager.
        
        Args:
            event_bus: Event bus for lifecycle events
        """
        self.event_bus = event_bus
        self.components: List[LifecycleComponent] = []
        self._logger = logging.getLogger(__name__)

    def add_component(self, component: LifecycleComponent) -> None:
        """Add a component to manage."""
        self.components.append(component)

    async def start_all(self) -> None:
        """Initialize and start all components."""
        # Initialize components
        for component in self.components:
            try:
                await component.initialize()
            except Exception as e:
                self._logger.error(f"Failed to initialize {component.name}: {str(e)}")
                await self._cleanup_initialized()
                raise

        # Start components
        started_components = []
        try:
            for component in self.components:
                await component.start()
                started_components.append(component)
        except Exception as e:
            self._logger.error(f"Failed to start components: {str(e)}")
            # Stop components that were started
            for component in reversed(started_components):
                try:
                    await component.stop()
                except Exception as stop_error:
                    self._logger.error(
                        f"Error stopping component {component.name}: {str(stop_error)}"
                    )
            raise

    async def stop_all(self) -> None:
        """Stop all components in reverse order."""
        for component in reversed(self.components):
            try:
                if component.state == ComponentState.RUNNING:
                    await component.stop()
            except Exception as e:
                self._logger.error(f"Error stopping component {component.name}: {str(e)}")
                await self.event_bus.publish(Event(
                    type=EventType.COMPONENT_ERROR,
                    data={
                        'component': component.name,
                        'error': str(e),
                        'operation': 'stop'
                    }
                ))

    async def cleanup_all(self) -> None:
        """Clean up all components in reverse order."""
        for component in reversed(self.components):
            try:
                await component.cleanup()
            except Exception as e:
                self._logger.error(f"Error cleaning up component {component.name}: {str(e)}")
                await self.event_bus.publish(Event(
                    type=EventType.COMPONENT_ERROR,
                    data={
                        'component': component.name,
                        'error': str(e),
                        'operation': 'cleanup'
                    }
                ))

    async def get_status(self) -> Dict[str, ComponentStatus]:
        """Get status of all components."""
        return {
            component.name: component._status
            for component in self.components
        }

    async def _cleanup_initialized(self) -> None:
        """Clean up components that were initialized."""
        for component in reversed(self.components):
            if component.state != ComponentState.CREATED:
                try:
                    await component.cleanup()
                except Exception as e:
                    self._logger.error(
                        f"Error cleaning up component {component.name}: {str(e)}"
                    )

    def __contains__(self, component_name: str) -> bool:
        """Check if a component is managed by this lifecycle."""
        return any(c.name == component_name for c in self.components)

    def get_component(self, name: str) -> Optional[LifecycleComponent]:
        """Get a component by name."""
        for component in self.components:
            if component.name == name:
                return component
        return None
```

```python
# core/context/embedding.py
"""Embedding management with lifecycle integration."""

import asyncio
import logging
import torch
from typing import Dict, List, Optional, Any
from datetime import datetime
from sentence_transformers import SentenceTransformer
from concurrent.futures import ThreadPoolExecutor
from core.events import EventBus, Event, EventType
from core.lifecycle import LifecycleComponent


class EmbeddingError(Exception):
    """Base exception for embedding-related errors."""
    pass


class EmbeddingManager(LifecycleComponent):
    """Manages text embeddings with lifecycle support."""

    def init(
        self,
        name: str,
        event_bus: EventBus,
        model_name: str = "all-MiniLM-L6-v2",
        batch_size: int = 32,
        cache_size: int = 1000
    ):
        """
        Initialize embedding manager.

        Args:
            name: Component name
            event_bus: Event bus for notifications
            model_name: Name of the embedding model
            batch_size: Size of batches for processing
            cache_size: Maximum number of cached embeddings
        """
        super().init(name, event_bus)
        self.model_name = model_name
        self.batch_size = batch_size
        self.cache_size = cache_size
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model: Optional[SentenceTransformer] = None
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._embedding_cache: Dict[str, torch.Tensor] = {}
        self._cache_usage: Dict[str, datetime] = {}
        self._lock = asyncio.Lock()

    async def do_initialize(self) -> None:
        """Initialize embedding model."""
        try:
            loop = asyncio.get_event_loop()
            self.model = await loop.run_in_executor(
                self._executor,
                lambda: SentenceTransformer(self.model_name).to(self.device)
            )
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_INITIALIZED,
                data={
                    'component': self.name,
                    'model': self.model_name,
                    'device': self.device
                }
            ))
        except Exception as e:
            raise EmbeddingError(f"Failed to initialize embedding model: {str(e)}") from e

    async def do_start(self) -> None:
        """Start embedding operations."""
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())

    async def do_stop(self) -> None:
        """Stop embedding operations."""
        if hasattr(self, '_cleanup_task'):
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

    async def do_cleanup(self) -> None:
        """Clean up embedding resources."""
        try:
            self._embedding_cache.clear()
            self._cache_usage.clear()

            if self.model:
                self.model = self.model.cpu()
                del self.model
                self.model = None

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            self._executor.shutdown(wait=True)

        except Exception as e:
            self._logger.error(f"Cleanup failed: {str(e)}")
            raise

    async def embed_text(self, text: str) -> torch.Tensor:
        """
        Generate embedding for text with caching.

        Args:
            text: Text to embed

        Returns:
            Embedding tensor

        Raises:
            EmbeddingError: If embedding generation fails
        """
        if not text.strip():
            raise EmbeddingError("Cannot embed empty text")

        try:
            cache_key = self._get_cache_key(text)
            async with self._lock:
                if cache_key in self._embedding_cache:
                    self._cache_usage[cache_key] = datetime.now()
                    return self._embedding_cache[cache_key]

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                self._executor,
                lambda: self.model.encode(
                    text,
                    convert_to_tensor=True,
                    device=self.device
                )
            )

            async with self._lock:
                if len(self._embedding_cache) >= self.cache_size:
                    await self._evict_cache_entry()

                self._embedding_cache[cache_key] = embedding
                self._cache_usage[cache_key] = datetime.now()

            return embedding

        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'embed_text'
                }
            ))
            raise EmbeddingError(f"Embedding generation failed: {str(e)}") from e

    async def embed_batch(
        self,
        texts: List[str],
        batch_size: Optional[int] = None
    ) -> List[torch.Tensor]:
        """
        Generate embeddings for multiple texts.

        Args:
            texts: List of texts to embed
            batch_size: Optional batch size override

        Returns:
            List of embedding tensors

        Raises:
            EmbeddingError: If batch embedding fails
        """
        if not texts:
            return []

        batch_size = batch_size or self.batch_size

        try:
            embeddings = []
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]

                batch_embeddings = []
                uncached_indices = []
                uncached_texts = []

                async with self._lock:
                    for j, text in enumerate(batch):
                        cache_key = self._get_cache_key(text)
                        if cache_key in self._embedding_cache:
                            batch_embeddings.append(self._embedding_cache[cache_key])
                            self._cache_usage[cache_key] = datetime.now()
                        else:
                            uncached_indices.append(j)
                            uncached_texts.append(text)

                if uncached_texts:
                    loop = asyncio.get_event_loop()
                    new_embeddings = await loop.run_in_executor(
                        self._executor,
                        lambda: self.model.encode(
                            uncached_texts,
                            convert_to_tensor=True,
                            device=self.device,
                            batch_size=batch_size
                        )
                    )

                    async with self._lock:
                        for text, embedding in zip(uncached_texts, new_embeddings):
                            cache_key = self._get_cache_key(text)
                            if len(self._embedding_cache) >= self.cache_size:
                                await self._evict_cache_entry()
                            self._embedding_cache[cache_key] = embedding
                            self._cache_usage[cache_key] = datetime.now()

                    for idx, embedding in zip(uncached_indices, new_embeddings):
                        batch_embeddings.insert(idx, embedding)

                embeddings.extend(batch_embeddings)

            return embeddings

        except Exception as e:
            await self.event_bus.publish(Event(
                type=EventType.COMPONENT_ERROR,
                data={
                    'component': self.name,
                    'error': str(e),
                    'operation': 'embed_batch'
                }
            ))
            raise EmbeddingError(f"Batch embedding failed: {str(e)}") from e

    async def get_similarity(self, embedding1: torch.Tensor, embedding2: torch.Tensor) -> float:
        """
        Calculate cosine similarity between embeddings.

        Args:
            embedding1: First embedding tensor
            embedding2: Second embedding tensor

        Returns:
            Similarity score between 0 and 1
        """
        try:
            similarity = torch.nn.functional.cosine_similarity(
                embedding1.unsqueeze(0),
                embedding2.unsqueeze(0)
            )
            return float(similarity.item())
        except Exception as e:
            raise EmbeddingError(f"Similarity calculation failed: {str(e)}") from e

    async def _cleanup_loop(self) -> None:
        """Background task to clean up old cache entries."""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                async with self._lock:
                    current_time = datetime.now()
                    expired_keys = [
                        key for key, last_used in self._cache_usage.items()
                        if (current_time - last_used).total_seconds() > 3600  # 1 hour expiry
                    ]

                    for key in expired_keys:
                        del self._embedding_cache[key]
                        del self._cache_usage[key]

                    if expired_keys:
                        await self.event_bus.publish(Event(
                            type=EventType.CACHE_UPDATED,
                            data={
                                'component': self.name,
                                'expired_entries': len(expired_keys)
                            }
                        ))

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._logger.error(f"Cache cleanup error: {str(e)}")

    async def _evict_cache_entry(self) -> None:
        """Evict least recently used cache entry."""
        if not self._cache_usage:
            return

        oldest_key = min(
            self._cache_usage.items(),
            key=lambda x: x[1]
        )[0]

        del self._embedding_cache[oldest_key]
        del self._cache_usage[oldest_key]

    def _get_cache_key(self, text: str) -> str:
        """Generate cache key for text."""
        import hashlib
        return hashlib.sha256(text.encode()).hexdigest()

    async def get_stats(self) -> Dict[str, Any]:
        """Get embedding statistics."""
        async with self._lock:
            return {
                'model': self.model_name,
                'device': self.device,
                'cache_size': len(self._embedding_cache),
                'max_cache_size': self.cache_size,
                'batch_size': self.batch_size,
                'cache_usage': {
                    key: last_used.isoformat()
                    for key, last_used in self._cache_usage.items()
                }
            }

    def __repr__(self) -> str:
        """String representation of embedding manager."""
        return (
            f"EmbeddingManager(model='{self.model_name}', "
            f"device='{self.device}', "
            f"cache_entries={len(self._embedding_cache)})"
        )
```

```python
# core/events/__init__.py
"""
Event System Module.

This module provides a robust event handling system for asyncio applications with 
support for typed events, priority handling, and error recovery.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum, auto
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, TypeVar, Generic
from uuid import UUID, uuid4

class EventPriority(Enum):
    """Event handling priority levels."""
    LOWEST = auto()
    LOW = auto()
    NORMAL = auto()
    HIGH = auto()
    HIGHEST = auto()
    MONITOR = auto()  # Special priority for monitoring/logging

class EventType(Enum):
    """Core system event types."""
    # Lifecycle events
    COMPONENT_INITIALIZED = auto()
    COMPONENT_STARTED = auto()
    COMPONENT_STOPPED = auto()
    COMPONENT_ERROR = auto()

    # Context events
    CONTEXT_UPDATED = auto()
    CONTEXT_OPTIMIZED = auto()
    CONTEXT_ERROR = auto()

    # Analysis events
    ANALYSIS_STARTED = auto()
    ANALYSIS_COMPLETED = auto()
    ANALYSIS_ERROR = auto()

    # Parser events
    PARSER_INITIALIZED = auto()
    PARSER_ERROR = auto()

    # Documentation events
    DOCUMENTATION_STARTED = auto()
    DOCUMENTATION_COMPLETED = auto()
    DOCUMENTATION_ERROR = auto()

    # Cache events
    CACHE_UPDATED = auto()
    CACHE_CLEARED = auto()
    CACHE_ERROR = auto()

@dataclass
class EventMetadata:
    """Metadata associated with an event."""
    event_id: UUID = field(default_factory=uuid4)
    timestamp: datetime = field(default_factory=datetime.now)
    source: Optional[str] = None
    correlation_id: Optional[str] = None
    causation_id: Optional[str] = None
    attributes: Dict[str, Any] = field(default_factory=dict)

T = TypeVar('T')

@dataclass
class Event(Generic[T]):
    """
    Base class for all events.
    
    Attributes:
        type: Event type
        data: Event payload
        metadata: Event metadata
        handled: Whether the event has been handled
        error: Optional error that occurred during handling
    """
    type: EventType
    data: T
    metadata: EventMetadata = field(default_factory=EventMetadata)
    handled: bool = False
    error: Optional[Exception] = None

@dataclass
class EventSubscription:
    """
    Represents a subscription to events.
    
    Attributes:
        callback: Async function to call when event occurs
        event_types: Event types to subscribe to
        priority: Handler priority
        filters: Optional additional filters
    """
    callback: Callable[[Event], Awaitable[None]]
    event_types: Set[EventType]
    priority: EventPriority = EventPriority.NORMAL
    filters: Optional[Dict[str, Any]] = None
    subscription_id: UUID = field(default_factory=uuid4)

class EventBus:
    """
    Central event dispatcher supporting async event handling.
    
    This class manages event subscriptions and handles event distribution
    to appropriate handlers based on event type and priority.
    """
    
    def __init__(self):
        """Initialize the event bus."""
        self._subscriptions: Dict[EventPriority, List[EventSubscription]] = {
            priority: [] for priority in EventPriority
        }
        self._handlers_lock = asyncio.Lock()
        self._logger = logging.getLogger(__name__)

    async def publish(self, event: Event) -> None:
        """
        Publish an event to all relevant subscribers.
        
        Args:
            event: Event to publish
        """
        try:
            # Get matching handlers by priority
            handlers = await self._get_matching_handlers(event)
            
            if not handlers:
                self._logger.debug(f"No handlers found for event: {event.type}")
                return
            
            # Execute handlers in priority order
            for priority in EventPriority:
                priority_handlers = [h for h in handlers if h.priority == priority]
                if priority_handlers:
                    await asyncio.gather(
                        *[self._execute_handler(handler, event) for handler in priority_handlers],
                        return_exceptions=True
                    )
                    
        except Exception as e:
            self._logger.error(f"Error publishing event: {str(e)}")
            raise

    async def subscribe(
        self,
        callback: Callable[[Event], Awaitable[None]],
        event_types: Set[EventType],
        priority: EventPriority = EventPriority.NORMAL,
        filters: Optional[Dict[str, Any]] = None
    ) -> EventSubscription:
        """
        Subscribe to events.
        
        Args:
            callback: Async function to call for matching events
            event_types: Set of event types to subscribe to
            priority: Handler priority
            filters: Optional additional filters
            
        Returns:
            Subscription object
        """
        subscription = EventSubscription(
            callback=callback,
            event_types=event_types,
            priority=priority,
            filters=filters
        )
        
        async with self._handlers_lock:
            self._subscriptions[priority].append(subscription)
            self._logger.debug(
                f"Registered handler for events {event_types} "
                f"with priority {priority.name}"
            )
            
        return subscription

    async def unsubscribe(self, subscription: EventSubscription) -> None:
        """
        Remove a subscription.
        
        Args:
            subscription: Subscription to remove
        """
        async with self._handlers_lock:
            self._subscriptions[subscription.priority] = [
                s for s in self._subscriptions[subscription.priority]
                if s.subscription_id != subscription.subscription_id
            ]

    async def _get_matching_handlers(self, event: Event) -> List[EventSubscription]:
        """Get handlers matching an event."""
        async with self._handlers_lock:
            matching = []
            for priority in EventPriority:
                for subscription in self._subscriptions[priority]:
                    if event.type in subscription.event_types:
                        if self._matches_filters(event, subscription.filters):
                            matching.append(subscription)
            return matching

    async def _execute_handler(self, subscription: EventSubscription, event: Event) -> None:
        """Execute a single event handler."""
        try:
            await subscription.callback(event)
            event.handled = True
        except Exception as e:
            self._logger.error(
                f"Error in event handler for {event.type}: {str(e)}",
                exc_info=True
            )
            event.error = e

    def _matches_filters(self, event: Event, filters: Optional[Dict[str, Any]]) -> bool:
        """Check if event matches additional filters."""
        if not filters:
            return True

        for key, value in filters.items():
            # Check metadata attributes
            if key.startswith('metadata.'):
                attr = key[9:]  # Remove 'metadata.' prefix
                if not hasattr(event.metadata, attr):
                    return False
                if getattr(event.metadata, attr) != value:
                    return False

            # Check data attributes
            elif key.startswith('data.'):
                attr = key[5:]  # Remove 'data.' prefix
                if not hasattr(event.data, attr):
                    return False
                if getattr(event.data, attr) != value:
                    return False

        return True
```