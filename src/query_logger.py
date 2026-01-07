"""
query_logger.py - Log all user queries, context, and responses
"""

import json
import pandas as pd
from datetime import datetime
from pathlib import Path
import hashlib


class QueryLogger:
    """Log user queries, AQL queries, and responses for analytics and debugging"""
    
    def __init__(self, log_dir="logs"):
        """Initialize logger with log directory"""
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.log_dir / "daily").mkdir(exist_ok=True)
        (self.log_dir / "failures").mkdir(exist_ok=True)
        
        # Current session logs
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_logs = []
    
    
    def log_query(self, 
                  user_question,
                  query_plan=None,
                  results=None,
                  llm_response=None,
                  execution_time=None,
                  error=None,
                  metadata=None):
        """
        Log a complete query interaction
        
        Args:
            user_question (str): Original user question
            query_plan (dict): Generated AQL query plan from LLM
            results (list): Query results from ArangoDB
            llm_response (str): Final LLM analysis/answer
            execution_time (float): Total execution time in seconds
            error (str): Error message if query failed
            metadata (dict): Additional context (user_id, session, etc.)
        """
        
        timestamp = datetime.now()
        
        # Generate unique query ID
        query_id = hashlib.md5(
            f"{timestamp.isoformat()}_{user_question}".encode()
        ).hexdigest()[:12]
        
        # Build log entry
        log_entry = {
            "query_id": query_id,
            "timestamp": timestamp.isoformat(),
            "session_id": self.session_id,
            
            # User input
            "user_question": user_question,
            "question_length": len(user_question),
            "question_word_count": len(user_question.split()),
            
            # Query planning
            "intent": query_plan.get("intent") if query_plan else None,
            "collections_used": query_plan.get("collections") if query_plan else None,
            "requires_embedding": query_plan.get("requires_embedding") if query_plan else False,
            "aql_query": query_plan.get("aql_query") if query_plan else None,
            "bind_vars": query_plan.get("bind_vars") if query_plan else None,
            "explanation": query_plan.get("explanation") if query_plan else None,
            
            # Execution results
            "success": error is None,
            "error": error,
            "result_count": len(results) if results else 0,
            "execution_time_seconds": execution_time,
            
            # LLM response
            "llm_response": llm_response,
            "response_length": len(llm_response) if llm_response else 0,
            
            # Additional metadata
            "metadata": metadata or {}
        }
        
        # Add to session logs
        self.session_logs.append(log_entry)
        
        # Write to daily log file (append mode)
        daily_log_file = self.log_dir / "daily" / f"queries_{timestamp.strftime('%Y%m%d')}.jsonl"
        with open(daily_log_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        # If query failed, also log to failures
        if error:
            failure_log_file = self.log_dir / "failures" / f"failures_{timestamp.strftime('%Y%m%d')}.jsonl"
            with open(failure_log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(log_entry, ensure_ascii=False) + '\n')
        
        return query_id
    
    
    def get_session_summary(self):
        """Get summary of current session"""
        if not self.session_logs:
            return {"message": "No queries logged in this session"}
        
        total_queries = len(self.session_logs)
        successful = sum(1 for log in self.session_logs if log['success'])
        failed = total_queries - successful
        
        avg_execution_time = sum(
            log.get('execution_time_seconds', 0) 
            for log in self.session_logs
        ) / total_queries if total_queries > 0 else 0
        
        collections_used = {}
        for log in self.session_logs:
            for coll in (log.get('collections_used') or []):
                collections_used[coll] = collections_used.get(coll, 0) + 1
        
        return {
            "session_id": self.session_id,
            "total_queries": total_queries,
            "successful": successful,
            "failed": failed,
            "success_rate": f"{(successful/total_queries*100):.1f}%" if total_queries > 0 else "N/A",
            "avg_execution_time": f"{avg_execution_time:.2f}s",
            "collections_used": collections_used,
            "intents": [log.get('intent') for log in self.session_logs if log.get('intent')]
        }
    
    
    def export_to_csv(self, output_file=None):
        """Export session logs to CSV for analysis"""
        if not self.session_logs:
            return None
        
        if output_file is None:
            output_file = self.log_dir / f"session_{self.session_id}.csv"
        
        # Flatten logs for CSV
        flattened = []
        for log in self.session_logs:
            flat_log = {
                "query_id": log["query_id"],
                "timestamp": log["timestamp"],
                "user_question": log["user_question"],
                "intent": log.get("intent"),
                "collections": ", ".join(log.get("collections_used") or []),
                "success": log["success"],
                "result_count": log["result_count"],
                "execution_time": log.get("execution_time_seconds"),
                "error": log.get("error"),
            }
            flattened.append(flat_log)
        
        df = pd.DataFrame(flattened)
        df.to_csv(output_file, index=False)
        
        return str(output_file)
    
    
    def get_failed_queries(self, days=7):
        """Get all failed queries from last N days"""
        failures = []
        
        for i in range(days):
            date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y%m%d')
            failure_file = self.log_dir / "failures" / f"failures_{date}.jsonl"
            
            if failure_file.exists():
                with open(failure_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        failures.append(json.loads(line))
        
        return failures
    
    
    def analyze_performance(self, days=7):
        """Analyze query performance over last N days"""
        all_logs = []
        
        for i in range(days):
            date = (datetime.now() - pd.Timedelta(days=i)).strftime('%Y%m%d')
            daily_file = self.log_dir / "daily" / f"queries_{date}.jsonl"
            
            if daily_file.exists():
                with open(daily_file, 'r', encoding='utf-8') as f:
                    for line in f:
                        all_logs.append(json.loads(line))
        
        if not all_logs:
            return {"message": "No logs found"}
        
        df = pd.DataFrame(all_logs)
        
        analysis = {
            "total_queries": len(df),
            "success_rate": f"{(df['success'].sum() / len(df) * 100):.1f}%",
            "avg_execution_time": f"{df['execution_time_seconds'].mean():.2f}s",
            "median_execution_time": f"{df['execution_time_seconds'].median():.2f}s",
            "avg_results_returned": f"{df['result_count'].mean():.1f}",
            
            "most_common_intents": df['intent'].value_counts().head(5).to_dict(),
            "most_used_collections": df['collections_used'].explode().value_counts().head(5).to_dict() if 'collections_used' in df else {},
            
            "slowest_queries": df.nlargest(5, 'execution_time_seconds')[
                ['user_question', 'execution_time_seconds', 'result_count']
            ].to_dict('records'),
            
            "most_common_errors": df[df['success'] == False]['error'].value_counts().head(5).to_dict() if not df[df['success'] == False].empty else {}
        }
        
        return analysis


# Global logger instance
_logger = None

def get_logger():
    """Get or create global logger instance"""
    global _logger
    if _logger is None:
        _logger = QueryLogger()
    return _logger
