# Data Engineering for Dejavu

**Pattern Extraction & Database Management for Cross-Similarity Forecasting**

**Date:** October 14, 2025  
**Model:** Dejavu - Data-Centric Forecasting  
**Objective:** Optimal pattern extraction and database management for production deployment

---

## Executive Summary

Data engineering for Dejavu focuses on:
1. **Pattern Extraction** - Converting time series into (pattern, outcome) pairs
2. **Pattern Database Management** - Efficient storage and retrieval
3. **Similarity Computation** - Fast distance calculations
4. **Online Learning** - Continuous database updates
5. **Quality Control** - Ensuring pattern coverage and relevance

**Critical Principle:** Database quality determines forecast quality - garbage in, garbage out!

---

## Table of Contents

1. [Pattern Extraction Pipeline](#pattern-extraction-pipeline)
2. [Database Management](#database-management)
3. [Similarity Preprocessing](#similarity-preprocessing)
4. [Online Learning](#online-learning)
5. [Quality Assurance](#quality-assurance)
6. [Implementation Code](#implementation-code)

---

## Pattern Extraction Pipeline

### 1. Time Series to Patterns

```python
class PatternExtractor:
    """
    Extract (pattern, outcome) pairs from time series
    """
    def __init__(
        self,
        pattern_length: int = 24,
        forecast_horizon: int = 1,
        stride: int = 1
    ):
        """
        Args:
            pattern_length: Length of historical pattern (h)
            forecast_horizon: Future steps to predict (H)
            stride: Step size for sliding window (1=every timestep)
        """
        self.h = pattern_length
        self.H = forecast_horizon
        self.stride = stride
    
    def extract(self, timeseries, timestamps=None):
        """
        Extract all valid (pattern, outcome) pairs
        
        Args:
            timeseries: 1D array of values
            timestamps: Optional timestamps for each value
        
        Returns:
            List of {'pattern', 'outcome', 'timestamp', 'metadata'}
        """
        patterns = []
        n = len(timeseries)
        
        for t in range(0, n - self.h - self.H + 1, self.stride):
            pattern = timeseries[t : t + self.h]
            outcome = timeseries[t + self.h : t + self.h + self.H]
            
            # Validate pattern (no NaN, sufficient variance)
            if self._is_valid_pattern(pattern, outcome):
                entry = {
                    'pattern': np.array(pattern, dtype=np.float32),
                    'outcome': np.array(outcome, dtype=np.float32),
                    'timestamp': timestamps[t] if timestamps is not None else t,
                    'pattern_end_idx': t + self.h - 1,
                    'metadata': {}
                }
                patterns.append(entry)
        
        print(f"Extracted {len(patterns)} patterns from {n} timesteps")
        return patterns
    
    def _is_valid_pattern(self, pattern, outcome):
        """Validate pattern quality"""
        # Check for NaN
        if np.isnan(pattern).any() or np.isnan(outcome).any():
            return False
        
        # Check for inf
        if np.isinf(pattern).any() or np.isinf(outcome).any():
            return False
        
        # Check variance (avoid constant patterns)
        if np.var(pattern) < 1e-10:
            return False
        
        return True
```

### 2. Multivariate Pattern Extraction

```python
class MultivariatePatternExtractor:
    """
    Extract patterns from multivariate time series
    """
    def __init__(self, pattern_length=24, forecast_horizon=1):
        self.h = pattern_length
        self.H = forecast_horizon
    
    def extract(self, df, feature_cols, target_col, time_col):
        """
        Extract multivariate patterns
        
        Args:
            df: DataFrame with time series
            feature_cols: List of feature column names
            target_col: Target column to predict
            time_col: Timestamp column
        
        Returns:
            List of pattern dictionaries
        """
        patterns = []
        n = len(df)
        
        for t in range(n - self.h - self.H + 1):
            # Multivariate pattern
            pattern_df = df[feature_cols].iloc[t : t + self.h]
            pattern = pattern_df.values  # Shape: (h, n_features)
            
            # Univariate or multivariate outcome
            outcome_df = df[target_col].iloc[t + self.h : t + self.h + self.H]
            outcome = outcome_df.values
            
            # Metadata (external conditions)
            metadata = df.iloc[t + self.h - 1].to_dict()
            
            patterns.append({
                'pattern': pattern.astype(np.float32),
                'outcome': outcome.astype(np.float32),
                'timestamp': df[time_col].iloc[t + self.h - 1],
                'metadata': metadata
            })
        
        return patterns
```

### 3. Conditional Pattern Extraction

```python
class ConditionalPatternExtractor:
    """
    Extract patterns with external conditions
    """
    def __init__(self, pattern_length=24, forecast_horizon=1):
        self.h = pattern_length
        self.H = forecast_horizon
    
    def extract_with_conditions(
        self,
        timeseries,
        conditions,
        timestamps=None
    ):
        """
        Extract patterns with associated conditions
        
        Args:
            timeseries: Main time series
            conditions: Dict of external variables
                e.g., {'temperature': [...], 'is_holiday': [...]}
            timestamps: Optional timestamps
        
        Returns:
            Patterns with condition metadata
        """
        patterns = []
        n = len(timeseries)
        
        for t in range(n - self.h - self.H + 1):
            pattern = timeseries[t : t + self.h]
            outcome = timeseries[t + self.h : t + self.h + self.H]
            
            # Extract conditions at forecast time
            pattern_conditions = {}
            for key, values in conditions.items():
                pattern_conditions[key] = values[t + self.h - 1]
            
            patterns.append({
                'pattern': np.array(pattern, dtype=np.float32),
                'outcome': np.array(outcome, dtype=np.float32),
                'timestamp': timestamps[t] if timestamps is not None else t,
                'conditions': pattern_conditions
            })
        
        return patterns
```

---

## Database Management

### 1. In-Memory Database

```python
class PatternDatabase:
    """
    In-memory pattern database with efficient operations
    """
    def __init__(self, max_size=10000):
        """
        Args:
            max_size: Maximum number of patterns (sliding window)
        """
        self.patterns = []
        self.max_size = max_size
        self.pattern_matrix = None  # For vectorized operations
        self.needs_rebuild = True
    
    def add_patterns(self, new_patterns):
        """Add multiple patterns"""
        self.patterns.extend(new_patterns)
        self._maintain_size()
        self.needs_rebuild = True
    
    def add_pattern(self, pattern_dict):
        """Add single pattern"""
        self.patterns.append(pattern_dict)
        self._maintain_size()
        self.needs_rebuild = True
    
    def _maintain_size(self):
        """Maintain max size with sliding window"""
        if len(self.patterns) > self.max_size:
            # Keep most recent patterns
            self.patterns = self.patterns[-self.max_size:]
    
    def build_matrix(self):
        """Build pattern matrix for vectorized operations"""
        if self.needs_rebuild and len(self.patterns) > 0:
            self.pattern_matrix = np.array([
                p['pattern'] for p in self.patterns
            ], dtype=np.float32)
            self.needs_rebuild = False
    
    def get_patterns(self):
        """Get all patterns"""
        return self.patterns
    
    def get_pattern_matrix(self):
        """Get pattern matrix (lazy build)"""
        if self.needs_rebuild:
            self.build_matrix()
        return self.pattern_matrix
    
    def __len__(self):
        return len(self.patterns)
    
    def save(self, filepath):
        """Save database to disk"""
        import joblib
        joblib.dump({
            'patterns': self.patterns,
            'max_size': self.max_size
        }, filepath, compress=3)
        print(f"Database saved: {len(self.patterns)} patterns → {filepath}")
    
    @classmethod
    def load(cls, filepath):
        """Load database from disk"""
        import joblib
        data = joblib.load(filepath)
        db = cls(max_size=data['max_size'])
        db.patterns = data['patterns']
        db.needs_rebuild = True
        print(f"Database loaded: {len(db.patterns)} patterns")
        return db
```

### 2. SQL-Based Database

```python
class SQLPatternDatabase:
    """
    Store patterns in SQL database for large-scale deployment
    """
    def __init__(self, connection_string):
        """
        Args:
            connection_string: Database connection string
        """
        from sqlalchemy import create_engine
        self.engine = create_engine(connection_string)
        self._create_tables()
    
    def _create_tables(self):
        """Create pattern storage tables"""
        create_table_sql = """
        CREATE TABLE IF NOT EXISTS patterns (
            id SERIAL PRIMARY KEY,
            pattern BYTEA NOT NULL,
            outcome BYTEA NOT NULL,
            timestamp TIMESTAMP,
            pattern_hash VARCHAR(32),
            metadata JSONB,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );
        
        CREATE INDEX IF NOT EXISTS idx_pattern_hash ON patterns(pattern_hash);
        CREATE INDEX IF NOT EXISTS idx_timestamp ON patterns(timestamp);
        """
        
        with self.engine.connect() as conn:
            conn.execute(create_table_sql)
    
    def add_patterns(self, patterns):
        """Add patterns to SQL database"""
        import hashlib
        import pickle
        
        records = []
        for p in patterns:
            pattern_bytes = pickle.dumps(p['pattern'])
            outcome_bytes = pickle.dumps(p['outcome'])
            pattern_hash = hashlib.md5(pattern_bytes).hexdigest()
            
            records.append({
                'pattern': pattern_bytes,
                'outcome': outcome_bytes,
                'timestamp': p.get('timestamp'),
                'pattern_hash': pattern_hash,
                'metadata': json.dumps(p.get('metadata', {}))
            })
        
        # Bulk insert
        from sqlalchemy import insert
        stmt = insert(patterns_table).values(records)
        
        with self.engine.connect() as conn:
            conn.execute(stmt)
        
        print(f"Added {len(records)} patterns to SQL database")
    
    def query_recent(self, limit=10000):
        """Get most recent patterns"""
        query = """
        SELECT pattern, outcome, timestamp, metadata
        FROM patterns
        ORDER BY timestamp DESC
        LIMIT %s
        """
        
        with self.engine.connect() as conn:
            result = conn.execute(query, (limit,))
            patterns = []
            
            for row in result:
                patterns.append({
                    'pattern': pickle.loads(row['pattern']),
                    'outcome': pickle.loads(row['outcome']),
                    'timestamp': row['timestamp'],
                    'metadata': json.loads(row['metadata'])
                })
        
        return patterns
```

### 3. Hybrid Storage (Memory + Disk)

```python
class HybridPatternDatabase:
    """
    Hot patterns in memory, cold patterns on disk
    """
    def __init__(self, hot_size=1000, cold_path='./cold_patterns/'):
        """
        Args:
            hot_size: Number of patterns to keep in memory
            cold_path: Directory for archived patterns
        """
        self.hot_db = PatternDatabase(max_size=hot_size)
        self.cold_path = Path(cold_path)
        self.cold_path.mkdir(exist_ok=True)
        self.cold_files = []
    
    def add_patterns(self, new_patterns):
        """Add patterns with automatic archiving"""
        self.hot_db.add_patterns(new_patterns)
        
        # Archive if hot db is full
        if len(self.hot_db) >= self.hot_db.max_size * 0.9:
            self._archive_cold_patterns()
    
    def _archive_cold_patterns(self):
        """Move old patterns to disk"""
        # Archive oldest 50% of hot patterns
        archive_count = len(self.hot_db) // 2
        patterns_to_archive = self.hot_db.patterns[:archive_count]
        
        # Save to disk
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filepath = self.cold_path / f'patterns_{timestamp}.pkl'
        
        import joblib
        joblib.dump(patterns_to_archive, filepath, compress=3)
        self.cold_files.append(filepath)
        
        # Remove from hot db
        self.hot_db.patterns = self.hot_db.patterns[archive_count:]
        self.hot_db.needs_rebuild = True
        
        print(f"Archived {archive_count} patterns to {filepath}")
    
    def load_cold_patterns(self, n_files=1):
        """Load recent cold patterns back to memory"""
        import joblib
        
        for filepath in self.cold_files[-n_files:]:
            patterns = joblib.load(filepath)
            self.hot_db.add_patterns(patterns)
        
        print(f"Loaded patterns from {n_files} cold files")
```

---

## Similarity Preprocessing

### 1. Normalization for Similarity

```python
class PatternNormalizer:
    """
    Normalize patterns for similarity computation
    """
    def __init__(self, method='zscore'):
        """
        Args:
            method: 'zscore', 'minmax', 'none'
        """
        self.method = method
    
    def normalize(self, pattern):
        """
        Normalize single pattern
        
        Returns:
            Normalized pattern, normalization parameters
        """
        pattern = np.array(pattern)
        
        if self.method == 'zscore':
            mean = np.mean(pattern)
            std = np.std(pattern)
            
            if std < 1e-10:
                return pattern, {'mean': mean, 'std': 1.0}
            
            normalized = (pattern - mean) / std
            return normalized, {'mean': mean, 'std': std}
        
        elif self.method == 'minmax':
            min_val = np.min(pattern)
            max_val = np.max(pattern)
            
            if max_val - min_val < 1e-10:
                return pattern, {'min': min_val, 'max': max_val}
            
            normalized = (pattern - min_val) / (max_val - min_val)
            return normalized, {'min': min_val, 'max': max_val}
        
        else:  # 'none'
            return pattern, {}
    
    def normalize_batch(self, patterns):
        """
        Normalize multiple patterns
        
        Returns:
            Normalized patterns, list of normalization params
        """
        normalized_patterns = []
        params_list = []
        
        for pattern in patterns:
            norm_pattern, params = self.normalize(pattern)
            normalized_patterns.append(norm_pattern)
            params_list.append(params)
        
        return np.array(normalized_patterns), params_list
```

### 2. Pattern Indexing for Fast Retrieval

```python
class PatternIndexer:
    """
    Index patterns for fast K-NN search
    """
    def __init__(self, method='kdtree'):
        """
        Args:
            method: 'kdtree', 'balltree', 'faiss', 'none'
        """
        self.method = method
        self.index = None
    
    def build_index(self, pattern_matrix):
        """
        Build index for pattern matrix
        
        Args:
            pattern_matrix: (n_patterns, pattern_length) array
        """
        if self.method == 'kdtree':
            from sklearn.neighbors import KDTree
            self.index = KDTree(pattern_matrix, metric='euclidean')
        
        elif self.method == 'balltree':
            from sklearn.neighbors import BallTree
            self.index = BallTree(pattern_matrix, metric='euclidean')
        
        elif self.method == 'faiss':
            import faiss
            d = pattern_matrix.shape[1]
            self.index = faiss.IndexFlatL2(d)
            self.index.add(pattern_matrix.astype('float32'))
        
        print(f"Index built: {self.method}, {len(pattern_matrix)} patterns")
    
    def query(self, query_pattern, k=10):
        """
        Find K nearest neighbors
        
        Returns:
            distances, indices
        """
        if self.method in ['kdtree', 'balltree']:
            distances, indices = self.index.query(
                query_pattern.reshape(1, -1),
                k=k
            )
            return distances[0], indices[0]
        
        elif self.method == 'faiss':
            distances, indices = self.index.search(
                query_pattern.reshape(1, -1).astype('float32'),
                k
            )
            return distances[0], indices[0]
        
        else:
            raise ValueError("No index built")
```

---

## Online Learning

### 1. Adaptive Database Updates

```python
class AdaptiveDatabaseManager:
    """
    Manage database updates with drift handling
    """
    def __init__(
        self,
        base_window_size=5000,
        min_window_size=1000,
        max_window_size=20000
    ):
        """
        Args:
            base_window_size: Initial sliding window size
            min_window_size: Minimum window size
            max_window_size: Maximum window size
        """
        self.window_size = base_window_size
        self.min_window = min_window_size
        self.max_window = max_window_size
        
        self.database = PatternDatabase(max_size=base_window_size)
        self.error_history = []
    
    def add_observation(self, pattern, outcome, actual_outcome=None):
        """
        Add new observation with adaptive windowing
        
        Args:
            pattern: Historical pattern
            outcome: What happened after pattern
            actual_outcome: True outcome (for validation)
        """
        # Add to database
        self.database.add_pattern({
            'pattern': pattern,
            'outcome': outcome,
            'timestamp': len(self.database)
        })
        
        # Track error if available
        if actual_outcome is not None:
            error = np.abs(outcome - actual_outcome).mean()
            self.error_history.append(error)
            
            # Adaptive windowing based on error
            if len(self.error_history) >= 100:
                self._adjust_window_size()
    
    def _adjust_window_size(self):
        """Adjust window size based on recent errors"""
        recent_errors = self.error_history[-100:]
        error_trend = np.mean(recent_errors[-20:]) / (np.mean(recent_errors[:20]) + 1e-10)
        
        # If error increasing (drift), reduce window
        if error_trend > 1.2:
            new_size = int(self.window_size * 0.8)
            self.window_size = max(new_size, self.min_window)
            print(f"Drift detected: window size → {self.window_size}")
        
        # If error stable/decreasing, can increase window
        elif error_trend < 0.9 and len(self.database) >= self.window_size:
            new_size = int(self.window_size * 1.1)
            self.window_size = min(new_size, self.max_window)
            print(f"Stable performance: window size → {self.window_size}")
        
        # Update database max size
        self.database.max_size = self.window_size
```

### 2. Streaming Pattern Extraction

```python
class StreamingPatternExtractor:
    """
    Extract patterns from streaming data
    """
    def __init__(self, pattern_length=24, forecast_horizon=1):
        self.h = pattern_length
        self.H = forecast_horizon
        self.buffer = []
        self.pending_outcomes = []
    
    def add_observation(self, value, timestamp=None):
        """
        Add new observation to stream
        
        Returns:
            New patterns if available
        """
        self.buffer.append({
            'value': value,
            'timestamp': timestamp or len(self.buffer)
        })
        
        new_patterns = []
        
        # Check if we can create new pattern
        if len(self.buffer) >= self.h + self.H:
            # Extract pattern
            pattern_values = [
                self.buffer[i]['value']
                for i in range(len(self.buffer) - self.H - self.h, len(self.buffer) - self.H)
            ]
            
            outcome_values = [
                self.buffer[i]['value']
                for i in range(len(self.buffer) - self.H, len(self.buffer))
            ]
            
            new_patterns.append({
                'pattern': np.array(pattern_values),
                'outcome': np.array(outcome_values),
                'timestamp': self.buffer[-self.H-1]['timestamp']
            })
        
        # Maintain buffer size
        max_buffer = self.h + self.H + 100
        if len(self.buffer) > max_buffer:
            self.buffer = self.buffer[-max_buffer:]
        
        return new_patterns
```

---

## Quality Assurance

### 1. Pattern Quality Validation

```python
class PatternQualityValidator:
    """
    Validate pattern database quality
    """
    def __init__(self):
        self.metrics = {}
    
    def validate_database(self, database):
        """
        Comprehensive database validation
        
        Returns:
            Dict of quality metrics
        """
        patterns = database.get_patterns()
        
        if len(patterns) == 0:
            return {'status': 'empty_database'}
        
        # Extract pattern and outcome arrays
        pattern_array = np.array([p['pattern'] for p in patterns])
        outcome_array = np.array([p['outcome'] for p in patterns])
        
        metrics = {}
        
        # 1. Size metrics
        metrics['n_patterns'] = len(patterns)
        metrics['pattern_length'] = pattern_array.shape[1]
        
        # 2. Coverage metrics
        metrics['pattern_mean'] = float(np.mean(pattern_array))
        metrics['pattern_std'] = float(np.std(pattern_array))
        metrics['pattern_min'] = float(np.min(pattern_array))
        metrics['pattern_max'] = float(np.max(pattern_array))
        
        # 3. Diversity metrics
        metrics['unique_patterns'] = self._count_unique_patterns(pattern_array)
        metrics['diversity_ratio'] = metrics['unique_patterns'] / metrics['n_patterns']
        
        # 4. Outcome statistics
        metrics['outcome_mean'] = float(np.mean(outcome_array))
        metrics['outcome_std'] = float(np.std(outcome_array))
        
        # 5. Quality checks
        metrics['has_nan'] = np.isnan(pattern_array).any() or np.isnan(outcome_array).any()
        metrics['has_inf'] = np.isinf(pattern_array).any() or np.isinf(outcome_array).any()
        
        # 6. Warnings
        warnings = []
        if metrics['n_patterns'] < 100:
            warnings.append("Low pattern count (<100)")
        if metrics['diversity_ratio'] < 0.5:
            warnings.append("Low diversity (many duplicate patterns)")
        if metrics['pattern_std'] < 0.1:
            warnings.append("Low variance in patterns")
        
        metrics['warnings'] = warnings
        metrics['quality_score'] = self._compute_quality_score(metrics)
        
        return metrics
    
    def _count_unique_patterns(self, pattern_array, tolerance=1e-6):
        """Count approximately unique patterns"""
        # Round patterns to reduce precision
        rounded = np.round(pattern_array / tolerance) * tolerance
        unique_count = len(np.unique(rounded, axis=0))
        return unique_count
    
    def _compute_quality_score(self, metrics):
        """Compute overall quality score 0-100"""
        score = 100.0
        
        # Penalize small database
        if metrics['n_patterns'] < 1000:
            score -= 20
        elif metrics['n_patterns'] < 100:
            score -= 40
        
        # Penalize low diversity
        if metrics['diversity_ratio'] < 0.7:
            score -= 20
        
        # Penalize data issues
        if metrics['has_nan'] or metrics['has_inf']:
            score -= 30
        
        return max(0.0, score)
```

### 2. Pattern Coverage Analysis

```python
def analyze_pattern_coverage(database, query_patterns, K=10):
    """
    Analyze how well database covers query patterns
    
    Args:
        database: Pattern database
        query_patterns: List of query patterns to test
        K: Number of neighbors
    
    Returns:
        Coverage statistics
    """
    from sklearn.neighbors import NearestNeighbors
    
    # Build K-NN
    db_patterns = database.get_pattern_matrix()
    knn = NearestNeighbors(n_neighbors=K, metric='euclidean')
    knn.fit(db_patterns)
    
    # Query coverage
    distances_list = []
    for query in query_patterns:
        distances, _ = knn.kneighbors(query.reshape(1, -1))
        distances_list.append(distances[0])
    
    distances_array = np.array(distances_list)
    
    coverage_stats = {
        'avg_nearest_distance': float(np.mean(distances_array[:, 0])),
        'avg_kth_distance': float(np.mean(distances_array[:, -1])),
        'max_nearest_distance': float(np.max(distances_array[:, 0])),
        'coverage_score': 100.0 / (1.0 + np.mean(distances_array[:, 0]))
    }
    
    return coverage_stats
```

---

## Implementation Code

### Complete Pipeline

```python
class DejavuDataPipeline:
    """
    Complete data engineering pipeline for Dejavu
    """
    def __init__(self, config):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Initialize components
        self.extractor = PatternExtractor(
            pattern_length=config['pattern_length'],
            forecast_horizon=config['forecast_horizon']
        )
        
        self.database = PatternDatabase(
            max_size=config.get('max_database_size', 10000)
        )
        
        self.normalizer = PatternNormalizer(
            method=config.get('normalization', 'zscore')
        )
        
        self.validator = PatternQualityValidator()
    
    def run(self, timeseries, timestamps=None):
        """
        Run complete pipeline
        
        Returns:
            Prepared database ready for forecasting
        """
        print("=" * 80)
        print("DEJAVU DATA PIPELINE")
        print("=" * 80)
        
        # Step 1: Extract patterns
        print("\n[1/4] Extracting patterns...")
        patterns = self.extractor.extract(timeseries, timestamps)
        print(f"  Extracted {len(patterns)} patterns")
        
        # Step 2: Normalize patterns
        print("\n[2/4] Normalizing patterns...")
        for p in patterns:
            p['pattern'], p['norm_params'] = self.normalizer.normalize(p['pattern'])
        print(f"  Normalized using {self.config.get('normalization', 'zscore')}")
        
        # Step 3: Add to database
        print("\n[3/4] Building database...")
        self.database.add_patterns(patterns)
        print(f"  Database size: {len(self.database)} patterns")
        
        # Step 4: Validate quality
        print("\n[4/4] Validating quality...")
        quality_metrics = self.validator.validate_database(self.database)
        print(f"  Quality score: {quality_metrics['quality_score']:.1f}/100")
        
        if quality_metrics['warnings']:
            print("  Warnings:")
            for warning in quality_metrics['warnings']:
                print(f"    - {warning}")
        
        print("\n" + "=" * 80)
        print("PIPELINE COMPLETE")
        print("=" * 80)
        
        return self.database, quality_metrics
```

### Usage Example

```python
# Configuration
config = {
    'pattern_length': 24,
    'forecast_horizon': 1,
    'max_database_size': 5000,
    'normalization': 'zscore'
}

# Load data
df = pd.read_csv('timeseries.csv')
timeseries = df['value'].values
timestamps = df['timestamp'].values

# Run pipeline
pipeline = DejavuDataPipeline(config)
database, metrics = pipeline.run(timeseries, timestamps)

# Save database
database.save('dejavu_database.pkl')

print(f"\n✓ Database ready: {len(database)} patterns")
print(f"✓ Quality score: {metrics['quality_score']:.1f}/100")
```

---

## Data Engineering Checklist

### Pattern Extraction
- [ ] Choose appropriate pattern length (domain seasonality)
- [ ] Set forecast horizon (prediction task)
- [ ] Validate pattern quality (no NaN, sufficient variance)
- [ ] Extract sufficient patterns (≥1000 recommended)
- [ ] Include temporal metadata

### Normalization
- [ ] Choose normalization method (zscore for amplitude-invariance)
- [ ] Normalize consistently (same method for all patterns)
- [ ] Store normalization parameters
- [ ] Validate normalized distributions

### Database Management
- [ ] Choose storage backend (memory, SQL, hybrid)
- [ ] Set appropriate max size (balance memory vs. coverage)
- [ ] Implement sliding window for non-stationarity
- [ ] Index for fast retrieval (if needed)
- [ ] Save/load functionality

### Quality Assurance
- [ ] Validate pattern coverage
- [ ] Check diversity (avoid too many duplicates)
- [ ] Monitor quality metrics over time
- [ ] Set up alerts for quality degradation
- [ ] Regular database cleanup

### Online Learning
- [ ] Implement streaming pattern extraction
- [ ] Set up periodic database updates
- [ ] Adaptive window sizing for drift
- [ ] Monitor forecast quality
- [ ] Recalibration triggers

---

## Best Practices

1. **Pattern Length Selection**
   - Energy (hourly): 24-168 (day-week)
   - Retail (daily): 7-28 (week-month)
   - Finance (minute): 60-300 (hour-session)

2. **Database Size**
   - Small: 1000-5000 (fast, limited coverage)
   - Medium: 5000-20000 (balanced)
   - Large: >20000 (slow, good coverage)

3. **Normalization**
   - Use z-score for amplitude-invariance
   - Use min-max for bounded domains
   - No normalization for scale-sensitive tasks

4. **Online Learning**
   - Update daily for stable domains
   - Update hourly for non-stationary
   - Sliding window size = 2-5x seasonality period

5. **Quality Monitoring**
   - Track coverage metrics weekly
   - Alert if quality score < 60
   - Rebuild database if drift detected

---

## Summary

**Data Engineering for Dejavu:**
- ✅ Pattern extraction from time series
- ✅ Quality validation and filtering
- ✅ Efficient database management
- ✅ Normalization for similarity
- ✅ Online learning and updates
- ✅ Quality monitoring

**Critical Success Factors:**
- Sufficient pattern coverage (≥1000 patterns)
- Appropriate pattern length (match domain seasonality)
- Quality patterns (no NaN, sufficient variance)
- Regular updates (sliding window for drift)
- Fast retrieval (indexing for large databases)

**Result:** High-quality pattern database ready for instant, interpretable forecasting!

---

**Ready for data-centric forecasting!** 📊

*Version 1.0.0 - October 14, 2025*

