"""
Shapley Value Attribution for Marketing Channels

This module implements Shapley value calculation for attributing conversion
value to marketing channels. It supports three methods:
- full: Complete Shapley value when all coalition data is available
- additive: Additive approximation when some coalitions are missing
- grouping: Uses average by coalition size when sample sizes are too small
"""

import math
from itertools import combinations
from collections import defaultdict
import pandas as pd


class ShapleyAttribution:
    """
    Compute Shapley value attribution for marketing channels.
    
    The Shapley value fairly distributes the total value among channels
    based on their marginal contributions across all possible coalitions.
    
    Attributes:
        conv_rate (dict): Mapping from frozenset of channels to conversion rate
        total_counts (dict): Mapping from frozenset of channels to sample size
        channels (list): List of channel names
        min_count (int): Minimum sample size to use actual data (default: 5)
        size_avg (dict): Average conversion rate by coalition size
    """
    
    def __init__(self, conv_rate, total_counts, channels, min_count=5):
        """
        Initialize ShapleyAttribution.
        
        Args:
            conv_rate (dict): Dictionary mapping frozenset of channels to conversion rate
            total_counts (dict): Dictionary mapping frozenset of channels to sample size
            channels (list): List of channel names
            min_count (int): Minimum sample size threshold for using actual data
        """
        self.conv_rate = conv_rate
        self.total_counts = total_counts
        self.channels = channels
        self.min_count = min_count
        self.size_avg = self._compute_avg_by_size()
    
    @classmethod
    def from_dataframe(cls, df, channels, rate_col='rate', 
                      size_col='sample_size', min_count=5):
        """
        Create ShapleyAttribution from a pandas DataFrame.
        
        The DataFrame should have:
        - One column per channel (0/1 indicating participation)
        - A column with conversion rates
        - A column with sample sizes
        
        Args:
            df (pd.DataFrame): DataFrame with channel columns and rate/size columns
            channels (list): List of channel names (must match DataFrame columns)
            rate_col (str): Name of conversion rate column
            size_col (str): Name of sample size column
            min_count (int): Minimum sample size threshold
            
        Returns:
            ShapleyAttribution: Initialized attribution object
        """
        conv_rate = {}
        total_counts = {}
        
        for _, row in df.iterrows():
            combo = frozenset([ch for ch in channels if row[ch] == 1])
            conv_rate[combo] = row[rate_col]
            total_counts[combo] = row[size_col]
        
        return cls(conv_rate, total_counts, channels, min_count)
    
    def _compute_avg_by_size(self):
        """
        Compute average conversion rate by coalition size.
        
        This is used for the grouping method when sample sizes are too small.
        
        Returns:
            dict: Mapping from coalition size to average conversion rate
        """
        size_groups = defaultdict(list)
        for combo, rate in self.conv_rate.items():
            size_groups[len(combo)].append(rate)
        return {size: sum(rates)/len(rates) for size, rates in size_groups.items() if rates}
    
    def _v(self, S, method):
        """
        Compute the value function v(S) for coalition S.
        
        Args:
            S (frozenset): Set of channels in the coalition
            method (str): Method to use ('full', 'additive', or 'grouping')
            
        Returns:
            float: Value of coalition S
        """
        k = len(S)
        S_fs = frozenset(S)
        
        if method == "full":
            # Use actual conversion rate if available, otherwise 0
            return self.conv_rate.get(S_fs, 0.0)
        
        elif method == "additive":
            # Additive approximation: base + sum of individual lifts
            base = self.conv_rate.get(frozenset(), 0.0)
            additive_value = base
            for c in S:
                individual_rate = self.conv_rate.get(frozenset([c]), base)
                additive_value += (individual_rate - base)
            return additive_value
        
        elif method == "grouping":
            # Use actual data if sample size is sufficient, otherwise use average by size
            if self.total_counts.get(S_fs, 0) >= self.min_count:
                return self.conv_rate.get(S_fs, 0.0)
            else:
                # Use average conversion rate for coalitions of this size
                return self.size_avg.get(k, 0.0)
        
        else:
            raise ValueError(f"Unsupported method: {method}. Use 'full', 'additive', or 'grouping'")
    
    def compute(self, method="auto"):
        """
        Compute Shapley values for all channels.
        
        The Shapley value for channel i is:
        φ_i = Σ_{S ⊆ N\{i}} [|S|!(n-|S|-1)!/n!] * [v(S ∪ {i}) - v(S)]
        
        Args:
            method (str): Method to use:
                - 'auto': Automatically select best method based on data availability
                - 'full': Use actual data for all coalitions
                - 'additive': Use additive approximation for missing data
                - 'grouping': Use grouping method for small sample sizes
        
        Returns:
            dict: Mapping from channel name to Shapley value
        """
        if method == "auto":
            method = self._select_method()
        
        n = len(self.channels)
        phi = {ch: 0.0 for ch in self.channels}
        N_fact = math.factorial(n)
        
        for ch in self.channels:
            rest = [c for c in self.channels if c != ch]
            for r in range(len(rest) + 1):
                for S in combinations(rest, r):
                    S_fs = frozenset(S)
                    v_S = self._v(S_fs, method)
                    v_S_ch = self._v(S_fs | {ch}, method)
                    weight = (math.factorial(r) * math.factorial(n - r - 1)) / N_fact
                    phi[ch] += weight * (v_S_ch - v_S)
        
        return phi
    
    def _select_method(self):
        """
        Automatically select the best method based on data availability.
        
        Returns:
            str: Selected method ('full', 'additive', or 'grouping')
        """
        n = len(self.channels)
        total_coalitions = 2 ** n
        
        # Check how many coalitions have data
        available_coalitions = len(self.conv_rate)
        coverage = available_coalitions / total_coalitions
        
        # Check how many have sufficient sample size
        sufficient_size = sum(1 for combo, count in self.total_counts.items() 
                            if count >= self.min_count)
        sufficient_coverage = sufficient_size / total_coalitions
        
        # Decision logic:
        # - If all coalitions have sufficient data, use 'full'
        # - If many coalitions missing but singles available, use 'additive'
        # - If data exists but sample sizes are small, use 'grouping'
        
        if coverage >= 0.9 and sufficient_coverage >= 0.9:
            return "full"
        elif coverage < 0.5:
            return "additive"
        else:
            return "grouping"
    
    def get_data_summary(self):
        """
        Get summary of available data.
        
        Returns:
            dict: Summary statistics about the data
        """
        n = len(self.channels)
        total_coalitions = 2 ** n
        available = len(self.conv_rate)
        sufficient_size = sum(1 for count in self.total_counts.values() 
                            if count >= self.min_count)
        
        return {
            'total_coalitions': total_coalitions,
            'available_coalitions': available,
            'coverage': available / total_coalitions,
            'sufficient_sample_size': sufficient_size,
            'sufficient_coverage': sufficient_size / total_coalitions,
            'min_count_threshold': self.min_count
        }
