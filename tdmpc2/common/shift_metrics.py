"""Population statistics for complete episodes and applied segments."""
import numpy as np


def summary_metrics(episodes):
    result = dict(episodes=len(episodes), std_ddof=0)
    for key in ('episode_return', 'success', 'episode_length', 'shift_events'):
        values = [row[key] for row in episodes]
        result[key + '_mean'] = float(np.mean(values)) if values else None
        result[key + '_std'] = float(np.std(values)) if values else None
    return result
