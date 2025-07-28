#!/usr/bin/env python3
"""
BAL Format Ticket Analyzer
Analyzes BAL format ticket data with specific rejection logic.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import argparse
import sys
from collections import Counter, defaultdict

class BALAnalyzer:
    def __init__(self, csv_file):
        self.csv_file = csv_file
        self.df = None
        self.approval_columns = []
        self.load_data()
    
    def load_data(self):
        """Load and prepare the BAL format ticket data"""
        try:
            self.df = pd.read_csv(self.csv_file)
            print(f"Loaded {len(self.df)} tickets from {self.csv_file}")
            
            # Store original values for analysis
            self.original_values = {}
            
            # Identify approval columns (all columns except RITM, Opened Date, Closed Date)
            all_columns = list(self.df.columns)
            self.approval_columns = [col for col in all_columns 
                                   if col not in ['RITM', 'Opened Date', 'Closed Date']]
            
            # Store original values before processing
            for col in self.approval_columns:
                self.original_values[col] = self.df[col].copy()
            
            # Process rejection logic: if BAL-INFO_SEC = NA, find rejection point
            self.process_rejections()
            
            # Convert timestamp columns to datetime
            timestamp_columns = ['Opened Date', 'Closed Date'] + self.approval_columns
            for col in timestamp_columns:
                if col in self.df.columns:
                    # Convert only non-NA values to datetime
                    mask = self.df[col] != 'NA'
                    if mask.any():
                        self.df.loc[mask, col] = pd.to_datetime(self.df.loc[mask, col], errors='coerce')
            
        except FileNotFoundError:
            print(f"Error: File {self.csv_file} not found")
            sys.exit(1)
        except Exception as e:
            print(f"Error loading data: {e}")
            sys.exit(1)
    
    def process_rejections(self):
        """Process rejection logic: if BAL-INFO_SEC = NA, find rejection point"""
        # Add Status and Rejected At columns
        self.df['Status'] = 'APPROVED'
        self.df['Rejected At'] = ''
        
        for idx, row in self.df.iterrows():
            # Check if BAL-INFO_SEC exists and is NA
            if 'BAL-INFO_SEC' in self.approval_columns:
                info_sec_value = row['BAL-INFO_SEC']
                if info_sec_value == 'NA':
                    # Find the last non-NA value (rejection point)
                    rejection_point = None
                    for col in self.approval_columns:
                        if row[col] != 'NA':
                            rejection_point = col
                    
                    if rejection_point:
                        # Mark as rejected
                        self.df.at[idx, 'Status'] = 'REJECTED'
                        self.df.at[idx, 'Rejected At'] = rejection_point
                        
                        # Mark all steps after rejection point as NOT NEEDED
                        rejection_index = self.approval_columns.index(rejection_point)
                        for i in range(rejection_index + 1, len(self.approval_columns)):
                            col = self.approval_columns[i]
                            self.df.at[idx, col] = 'NOT NEEDED'
                        
                        # Set Closed Date to rejection timestamp
                        if pd.notna(self.df.at[idx, rejection_point]):
                            self.df.at[idx, 'Closed Date'] = self.df.at[idx, rejection_point]
    
    def calculate_step_durations(self):
        """Calculate time spent in each approval step"""
        durations = {}
        
        for i, col in enumerate(self.approval_columns):
            step_name = col
            
            if i == 0:
                # First approval: from Opened Date to first approval
                start_col = 'Opened Date'
            else:
                # Other approvals: from previous approval to current approval
                start_col = self.approval_columns[i-1]
            
            # Calculate duration only for completed approvals (not NA, not NOT NEEDED)
            mask = (self.df[col] != 'NA') & (self.df[col] != 'NOT NEEDED') & (self.df[col].notna())
            if mask.sum() > 0:
                # Ensure both columns are datetime before calculation
                col_data = self.df.loc[mask, col]
                start_data = self.df.loc[mask, start_col]
                if pd.api.types.is_datetime64_any_dtype(col_data) and pd.api.types.is_datetime64_any_dtype(start_data):
                    duration = col_data - start_data
                    durations[step_name] = duration
                else:
                    durations[step_name] = pd.Series(dtype='timedelta64[ns]')
            else:
                durations[step_name] = pd.Series(dtype='timedelta64[ns]')
        
        return durations
    
    def analyze_timing_statistics(self):
        """Analyze timing statistics for each approval step"""
        print("\n" + "="*80)
        print("APPROVAL STEP TIMING ANALYSIS")
        print("="*80)
        
        durations = self.calculate_step_durations()
        
        for step_name, duration_series in durations.items():
            if len(duration_series) == 0:
                print(f"\n{step_name}:")
                print("  No completed approvals found")
                continue
            
            # Convert to days for primary interpretation
            days = duration_series.dt.total_seconds() / 86400
            hours = duration_series.dt.total_seconds() / 3600
            
            print(f"\n{step_name}:")
            print(f"  Completed approvals: {len(duration_series)}")
            print(f"  Mean time: {days.mean():.2f} days ({hours.mean():.2f} hours)")
            print(f"  Median time: {days.median():.2f} days ({hours.median():.2f} hours)")
            print(f"  Standard deviation: {days.std():.2f} days ({hours.std():.2f} hours)")
            print(f"  Min time: {days.min():.2f} days ({hours.min():.2f} hours)")
            print(f"  Max time: {days.max():.2f} days ({hours.max():.2f} hours)")
    
    def analyze_skip_patterns(self):
        """Analyze which approvals are most commonly skipped"""
        print("\n" + "="*80)
        print("SKIP PATTERNS ANALYSIS")
        print("="*80)
        
        for col in self.approval_columns:
            total_tickets = len(self.df)
            na_count = (self.df[col] == 'NA').sum()
            not_needed_count = (self.df[col] == 'NOT NEEDED').sum()
            completed_count = total_tickets - na_count - not_needed_count
            
            na_percent = (na_count / total_tickets) * 100
            not_needed_percent = (not_needed_count / total_tickets) * 100
            completed_percent = (completed_count / total_tickets) * 100
            
            print(f"\n{col}:")
            print(f"  Completed: {completed_count} ({completed_percent:.1f}%)")
            print(f"  Skipped (NA): {na_count} ({na_percent:.1f}%)")
            print(f"  Not needed (rejected): {not_needed_count} ({not_needed_percent:.1f}%)")
    
    def analyze_approval_paths(self):
        """Analyze common approval workflow patterns"""
        print("\n" + "="*80)
        print("APPROVAL PATH ANALYSIS")
        print("="*80)
        
        paths = []
        for _, row in self.df.iterrows():
            path = []
            for col in self.approval_columns:
                value = row[col]
                if value == 'NA':
                    path.append('S')  # Skipped
                elif value == 'NOT NEEDED':
                    path.append('N')  # Not needed due to rejection
                elif pd.notna(value):
                    path.append('C')  # Completed
                else:
                    path.append('S')  # Default to skipped
            
            paths.append(''.join(path))
        
        # Count path frequencies
        path_counts = Counter(paths)
        
        print(f"\nTotal unique paths: {len(path_counts)}")
        print(f"Most common approval paths:")
        
        for path, count in path_counts.most_common(10):
            percentage = (count / len(self.df)) * 100
            description = self.describe_path(path)
            print(f"  {path}: {count} tickets ({percentage:.1f}%) - {description}")
        
        print(f"\nPath Legend:")
        print(f"  C = Completed (✓)")
        print(f"  S = Skipped (○)")
        print(f"  N = Not needed due to rejection (✗)")
    
    def describe_path(self, path):
        """Describe what a path signature means"""
        if len(path) != len(self.approval_columns):
            return "Invalid path length"
        
        description = []
        for i, step in enumerate(path):
            step_name = self.approval_columns[i]
            if step == 'C':
                description.append(f"{step_name}: ✓")
            elif step == 'S':
                description.append(f"{step_name}: ○")
            elif step == 'N':
                description.append(f"{step_name}: ✗")
        
        return " → ".join(description)
    
    def analyze_total_processing_time(self):
        """Analyze total processing time from open to close"""
        print("\n" + "="*80)
        print("TOTAL PROCESSING TIME ANALYSIS")
        print("="*80)
        
        # Calculate total processing time
        mask = (self.df['Opened Date'].notna()) & (self.df['Closed Date'].notna())
        if mask.sum() == 0:
            print("No valid open/close dates found")
            return
        
        total_time = self.df.loc[mask, 'Closed Date'] - self.df.loc[mask, 'Opened Date']
        # Ensure we have datetime values before using .dt accessor
        if pd.api.types.is_timedelta64_dtype(total_time):
            days = total_time.dt.total_seconds() / 86400
            hours = total_time.dt.total_seconds() / 3600
        else:
            print("No valid datetime calculations possible")
            return
        
        print(f"Total tickets with valid dates: {len(days)}")
        print(f"Mean processing time: {days.mean():.2f} days ({hours.mean():.2f} hours)")
        print(f"Median processing time: {days.median():.2f} days ({hours.median():.2f} hours)")
        print(f"Standard deviation: {days.std():.2f} days ({hours.std():.2f} hours)")
        print(f"Min processing time: {days.min():.2f} days ({hours.min():.2f} hours)")
        print(f"Max processing time: {days.max():.2f} days ({hours.max():.2f} hours)")
        
        # Breakdown by approval count
        print(f"\nProcessing time by number of approvals:")
        for _, row in self.df.loc[mask].iterrows():
            approval_count = sum(1 for col in self.approval_columns 
                               if row[col] != 'NA' and row[col] != 'NOT NEEDED' and pd.notna(row[col]))
            print(f"  {approval_count} approvals: {days.iloc[0]:.2f} days")
    
    def analyze_bottlenecks(self):
        """Identify bottlenecks in the approval process"""
        print("\n" + "="*80)
        print("BOTTLENECK ANALYSIS")
        print("="*80)
        
        durations = self.calculate_step_durations()
        
        # Calculate average processing time for each step
        step_times = {}
        for step_name, duration_series in durations.items():
            if len(duration_series) > 0:
                days = duration_series.dt.total_seconds() / 86400
                step_times[step_name] = {
                    'mean': days.mean(),
                    'median': days.median(),
                    'std': days.std(),
                    'count': len(duration_series)
                }
        
        # Sort by average processing time
        sorted_steps = sorted(step_times.items(), key=lambda x: x[1]['mean'], reverse=True)
        
        print("Steps ranked by average processing time:")
        for i, (step_name, stats) in enumerate(sorted_steps, 1):
            print(f"  {i}. {step_name}: {stats['mean']:.2f} days (median: {stats['median']:.2f}, std: {stats['std']:.2f}, count: {stats['count']})")
    
    def analyze_rejections(self):
        """Analyze rejection patterns"""
        print("\n" + "="*80)
        print("REJECTION ANALYSIS")
        print("="*80)
        
        rejected_tickets = self.df[self.df['Status'] == 'REJECTED']
        total_tickets = len(self.df)
        
        print(f"Total tickets: {total_tickets}")
        print(f"Rejected tickets: {len(rejected_tickets)} ({(len(rejected_tickets)/total_tickets)*100:.1f}%)")
        print(f"Approved tickets: {total_tickets - len(rejected_tickets)} ({((total_tickets - len(rejected_tickets))/total_tickets)*100:.1f}%)")
        
        if len(rejected_tickets) > 0:
            print(f"\nRejection points:")
            rejection_counts = rejected_tickets['Rejected At'].value_counts()
            for step, count in rejection_counts.items():
                percentage = (count / len(rejected_tickets)) * 100
                print(f"  {step}: {count} rejections ({percentage:.1f}%)")
            
            # Analyze processing time for rejected tickets
            mask = (rejected_tickets['Opened Date'].notna()) & (rejected_tickets['Closed Date'].notna())
            if mask.sum() > 0:
                rejection_time = rejected_tickets.loc[mask, 'Closed Date'] - rejected_tickets.loc[mask, 'Opened Date']
                days = rejection_time.dt.total_seconds() / 86400
                print(f"\nRejection processing time:")
                print(f"  Mean: {days.mean():.2f} days")
                print(f"  Median: {days.median():.2f} days")
                print(f"  Min: {days.min():.2f} days")
                print(f"  Max: {days.max():.2f} days")
    
    def generate_summary_report(self):
        """Generate a comprehensive summary report"""
        print("\n" + "="*80)
        print("SUMMARY REPORT")
        print("="*80)
        
        total_tickets = len(self.df)
        approved_tickets = len(self.df[self.df['Status'] == 'APPROVED'])
        rejected_tickets = len(self.df[self.df['Status'] == 'REJECTED'])
        
        # Calculate average approvals per ticket
        approval_counts = []
        for _, row in self.df.iterrows():
            count = sum(1 for col in self.approval_columns 
                       if row[col] != 'NA' and row[col] != 'NOT NEEDED' and pd.notna(row[col]))
            approval_counts.append(count)
        
        avg_approvals = np.mean(approval_counts)
        
        # Date range
        if self.df['Opened Date'].notna().any():
            date_range = (self.df['Opened Date'].max() - self.df['Opened Date'].min()).days
        else:
            date_range = 0
        
        print(f"Dataset Overview:")
        print(f"  Total tickets: {total_tickets}")
        print(f"  Approved tickets: {approved_tickets}")
        print(f"  Rejected tickets: {rejected_tickets}")
        print(f"  Approval rate: {(approved_tickets/total_tickets)*100:.1f}%")
        print(f"  Date range: {date_range} days")
        print(f"  Average approvals per ticket: {avg_approvals:.1f}")
        
        # Most common path
        paths = []
        for _, row in self.df.iterrows():
            path = []
            for col in self.approval_columns:
                value = row[col]
                if value == 'NA':
                    path.append('S')
                elif value == 'NOT NEEDED':
                    path.append('N')
                elif pd.notna(value):
                    path.append('C')
                else:
                    path.append('S')
            paths.append(''.join(path))
        
        if paths:
            most_common_path = Counter(paths).most_common(1)[0][0]
            print(f"  Most common approval path: {most_common_path}")
    
    def run_full_analysis(self):
        """Run the complete analysis suite"""
        self.generate_summary_report()
        self.analyze_timing_statistics()
        self.analyze_skip_patterns()
        self.analyze_approval_paths()
        self.analyze_total_processing_time()
        self.analyze_bottlenecks()
        self.analyze_rejections()

def main():
    parser = argparse.ArgumentParser(description='Analyze BAL format ticket data')
    parser.add_argument('csv_file', help='Path to the CSV file containing ticket data')
    parser.add_argument('--summary', action='store_true', help='Show summary report only')
    parser.add_argument('--timing', action='store_true', help='Show timing analysis only')
    parser.add_argument('--skips', action='store_true', help='Show skip patterns only')
    parser.add_argument('--paths', action='store_true', help='Show approval paths only')
    parser.add_argument('--bottlenecks', action='store_true', help='Show bottleneck analysis only')
    parser.add_argument('--rejections', action='store_true', help='Show rejection analysis only')
    
    args = parser.parse_args()
    
    analyzer = BALAnalyzer(args.csv_file)
    
    if args.summary:
        analyzer.generate_summary_report()
    elif args.timing:
        analyzer.analyze_timing_statistics()
    elif args.skips:
        analyzer.analyze_skip_patterns()
    elif args.paths:
        analyzer.analyze_approval_paths()
    elif args.bottlenecks:
        analyzer.analyze_bottlenecks()
    elif args.rejections:
        analyzer.analyze_rejections()
    else:
        analyzer.run_full_analysis()

if __name__ == "__main__":
    main() 