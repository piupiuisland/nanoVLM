#!/usr/bin/env python3
import json
import os
import sys
import glob
import argparse
import matplotlib.pyplot as plt
import pandas as pd

METRIC_TITLE_MAPPING = {
        'docvqa_val_anls': 'DocVQA',
        'infovqa_val_anls': 'InfoVQA',
        'mme_total_score': 'MME Total',
        'mmmu_val_mmmu_acc': 'MMMU',
        'mmstar_average': 'MMStar',
        'ocrbench_ocrbench_accuracy': 'OCRBench',
        'scienceqa_exact_match': 'ScienceQA',
        'textvqa_val_exact_match': 'TextVQA',
        'average': 'Average',
        'average_rank': 'Average Rank',
        'ai2d_exact_match': 'AI2D',
        'chartqa_relaxed_overall': 'ChartQA',
        'seedbench_seed_all': 'SeedBench'
    }

def compute_ranking_summary(all_results, tasks_to_plot):
    """Compute ranking-based summary metric across all runs."""
    if not all_results or len(all_results) < 2:
        return all_results
    
    # Get all steps that appear in all runs
    all_steps = set()
    for results in all_results:
        all_steps.update(result['step'] for result in results)
    
    # For each step, compute rankings
    for step in all_steps:
        # Find all runs that have this step
        step_data = []
        run_indices = []
        
        for run_idx, results in enumerate(all_results):
            step_result = next((r for r in results if r['step'] == step), None)
            if step_result:
                step_data.append(step_result)
                run_indices.append(run_idx)
        
        if len(step_data) < 2:
            continue
            
        # Get metrics to rank (exclude 'average' and 'average_rank' from ranking calculation)
        metrics_to_rank = []
        if tasks_to_plot:
            for task in tasks_to_plot:
                if task not in ['average', 'average_rank'] and task in step_data[0]:
                    metrics_to_rank.append(task)
        else:
            metrics_to_rank = [k for k in step_data[0].keys() if k not in ['step', 'average', 'average_rank']]
        
        if not metrics_to_rank:
            continue
            
        # Compute rankings for each metric
        rankings = []
        for metric in metrics_to_rank:
            # Get values for this metric across all runs at this step
            metric_values = []
            for data in step_data:
                if metric in data and isinstance(data[metric], (int, float)):
                    metric_values.append(data[metric])
                else:
                    metric_values.append(None)
            
            # Skip this metric if any run is missing it
            if None in metric_values:
                continue
                
            # Create ranking (higher value = better rank, so we rank in descending order)
            # Convert to list of (value, original_index) pairs
            indexed_values = [(val, idx) for idx, val in enumerate(metric_values)]
            # Sort by value in descending order (higher is better)
            indexed_values.sort(key=lambda x: x[0], reverse=True)
            
            # Assign ranks (1 is best)
            metric_rankings = [0] * len(metric_values)
            for rank, (_, original_idx) in enumerate(indexed_values, 1):
                metric_rankings[original_idx] = rank
                
            rankings.append(metric_rankings)
        
        # Compute average ranking for each run
        if rankings:
            avg_rankings = []
            for run_idx in range(len(step_data)):
                run_ranks = [ranking[run_idx] for ranking in rankings]
                avg_rankings.append(sum(run_ranks) / len(run_ranks))
            
            # Add ranking summary to each run's data
            for i, (data, run_idx) in enumerate(zip(step_data, run_indices)):
                # Find the result in the original data and add ranking summary
                for result in all_results[run_idx]:
                    if result['step'] == step:
                        result['average_rank'] = avg_rankings[i]
                        break
    
    return all_results

def load_eval_results(eval_folder, tasks_to_plot=None):
    """Load all JSON files from the evaluation folder and extract results."""
    json_files = glob.glob(os.path.join(eval_folder, "step_*.json"))
    
    if not json_files:
        print(f"No JSON files found in {eval_folder}")
        return None
    
    results = []
    for json_file in json_files:
        with open(json_file, 'r') as f:
            data = json.load(f)
            step = data.get('global_step', 0)
            metrics = data.get('results', {})
            
            result = {'step': step}
            result.update(metrics)
            
            # Add MME total score if mme is in tasks and both perception and cognition scores exist
            if tasks_to_plot and any('mme_total_score' in task.lower() for task in tasks_to_plot):
                perception_score = result.get('mme_mme_perception_score')
                cognition_score = result.get('mme_mme_cognition_score')
                
                if perception_score is not None and cognition_score is not None:
                    result['mme_total_score'] = perception_score + cognition_score
            
            # Add average score if 'average' is in tasks
            if tasks_to_plot and 'average' in tasks_to_plot:
                # Get only the specified tasks (excluding 'average')
                metrics_to_average = []
                for task in tasks_to_plot:
                    if (task != 'average' and 
                        'rank' not in task.lower() and
                        task in result and
                        isinstance(result[task], (int, float))):
                        # Special handling for MME total score: normalize by dividing by 2800
                        if task == 'mme_total_score':
                            normalized_score = result[task] / 2800.0
                            metrics_to_average.append(normalized_score)
                        else:
                            metrics_to_average.append(result[task])
                
                if metrics_to_average:
                    result['average'] = sum(metrics_to_average) / len(metrics_to_average)
            
            results.append(result)
    
    # Sort by step
    results.sort(key=lambda x: x['step'])
    return results

def get_legend_name(eval_folder, custom_name=None):
    """Extract legend name from folder path or use custom name."""
    if custom_name:
        return custom_name
    folder_name = os.path.basename(eval_folder)
    return folder_name.split('_')[-1]

def plot_results(all_results, eval_folders, custom_names=None, tasks_to_plot=None, output_filename=None, steps_to_plot=None):
    """Plot the evaluation results for multiple folders."""
    if not all_results:
        return
    
    # Set academic style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['mathtext.fontset'] = 'cm'
    
    # Mapping from metric names to display titles
    
    # Extract all metric names from all results
    metric_names = set()
    for results in all_results:
        for result in results:
            metric_names.update(k for k in result.keys() if k != 'step')
    
    # Filter metrics based on specified tasks if provided
    if tasks_to_plot:
        filtered_metrics = set()
        for task in tasks_to_plot:
            # Exact match for specified tasks
            if task in metric_names:
                filtered_metrics.add(task)
        metric_names = filtered_metrics
        
        if not metric_names:
            print(f"Warning: No metrics found exactly matching tasks: {tasks_to_plot}")
            return
    
    metric_names = sorted(list(metric_names))
    
    # Create subplots
    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols
    
    _, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Define academic colors and markers for different runs
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
    
    for i, metric in enumerate(metric_names):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        # Plot each run
        for j, (results, eval_folder) in enumerate(zip(all_results, eval_folders)):
            # Extract values for this metric
            values = []
            metric_steps = []
            missing_steps = []
            
            for result in results:
                # Check if we should include this step
                if steps_to_plot is None or result['step'] in steps_to_plot:
                    if metric in result:
                        values.append(result[metric])
                        metric_steps.append(result['step'])
                    elif steps_to_plot is not None:
                        # Only log missing if specific steps were requested
                        missing_steps.append(result['step'])
            
            # Log missing metrics for specified steps
            if missing_steps:
                folder_name = custom_names[j] if custom_names and custom_names[j] else os.path.basename(eval_folder)
                print(f"Warning: {folder_name} missing '{metric}' for steps: {missing_steps}")
            
            if values:
                custom_name = custom_names[j] if custom_names else None
                legend_name = get_legend_name(eval_folder, custom_name)
                color = colors[j % len(colors)]
                marker = markers[j % len(markers)]
                
                # Check if there's stderr data for this metric
                stderr_metric = metric + '_stderr'
                stderr_values = []
                for result in results:
                    if steps_to_plot is None or result['step'] in steps_to_plot:
                        if metric in result and stderr_metric in result:
                            stderr_values.append(result[stderr_metric])
                        elif metric in result:
                            stderr_values.append(0)  # No stderr available for this step
                
                # Plot the main line
                ax.plot(metric_steps, values, marker=marker, markersize=4, 
                       color=color, label=legend_name, linewidth=2, alpha=0.9)
                
                # Plot error corridor if stderr data is available
                if stderr_values and any(stderr > 0 for stderr in stderr_values):
                    lower_bounds = [v - s for v, s in zip(values, stderr_values)]
                    upper_bounds = [v + s for v, s in zip(values, stderr_values)]
                    ax.fill_between(metric_steps, lower_bounds, upper_bounds, 
                                  color=color, alpha=0.2, linewidth=0)
        
        # Get display title from mapping, fallback to original metric name
        display_title = METRIC_TITLE_MAPPING.get(metric, metric)
        ax.set_title(display_title, fontsize=13, weight='bold')
        ax.set_xlabel('Training Step (×1000)', fontsize=10, weight='bold')
        ax.set_ylabel('Value', fontsize=11, weight='bold')
        ax.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
        
        # Set x-axis ticks to show simple integers (steps divided by 1000)
        # Get all steps for this metric across all runs
        all_metric_steps = []
        for results in all_results:
            for result in results:
                if (steps_to_plot is None or result['step'] in steps_to_plot) and metric in result:
                    all_metric_steps.append(result['step'])
        
        if all_metric_steps:
            unique_steps = sorted(set(all_metric_steps))
            ax.set_xticks(unique_steps)
            # Show only every second label to avoid crowding
            labels = [int(step/1000) if i % 2 == 0 else '' for i, step in enumerate(unique_steps)]
            ax.set_xticklabels(labels)
        
        # Invert y-axis for ranking metrics (lower rank = better performance)
        if 'rank' in metric.lower():
            ax.invert_yaxis()
        
        # Add subtle background and improve spines
        ax.set_facecolor('#fafafa')
        for spine in ax.spines.values():
            spine.set_linewidth(1.2)
            spine.set_color('#333333')
        
        if len(eval_folders) > 1:
            ax.legend(loc='upper left', frameon=True, fancybox=True, shadow=False,
                     framealpha=0.9, edgecolor='gray', fontsize=10)
    
    # Hide unused subplots
    for i in range(n_metrics, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    # # Add title if output filename is specified
    # if output_filename:
    #     plt.suptitle(output_filename, fontsize=16, y=0.98)
    
    plt.tight_layout()
    
    # Create assets folder if it doesn't exist
    assets_folder = '/fsx/luis_wiedmann/nanoVLM/plots_final'
    os.makedirs(assets_folder, exist_ok=True)
    
    # Save the plot to assets folder
    if output_filename:
        output_file = os.path.join(assets_folder, f'{output_filename}.pdf')
    elif len(eval_folders) == 1:
        folder_name = os.path.basename(eval_folders[0])
        output_file = os.path.join(assets_folder, f'{folder_name}_evaluation_plots.pdf')
    else:
        output_file = os.path.join(assets_folder, 'comparison_evaluation_plots.pdf')
    
    plt.savefig(output_file, format='pdf', dpi=600, bbox_inches='tight')
    print(f"Plot saved to: {output_file}")
    
    plt.close()
    
    # Save individual plots as PDFs for specified metrics
    individual_plots = ['average_rank', 'average']  # Add more metrics here as needed
    for metric in individual_plots:
        if metric in metric_names:
            save_individual_plot_pdf(all_results, eval_folders, custom_names, output_filename, metric, steps_to_plot)
    
    # Save CSV data
    save_csv_data(all_results, eval_folders, custom_names, metric_names, output_file, steps_to_plot)

def save_individual_plot_pdf(all_results, eval_folders, custom_names, output_filename, metric_name, steps_to_plot=None):
    """Save an individual metric plot as a PDF with 300 DPI and no title."""
    # Set academic style
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 12
    plt.rcParams['mathtext.fontset'] = 'cm'
    
    # Create a new figure with golden ratio proportions
    #plt.figure(figsize=(10, 6.18))
    
    # Define academic colors and markers
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    markers = ['o', 's', '^', 'D', 'v', 'p', '*', 'h', '+', 'x']
    
    # Plot each run for the specified metric
    for j, (results, eval_folder) in enumerate(zip(all_results, eval_folders)):
        # Extract values for this metric
        values = []
        metric_steps = []
        
        for result in results:
            # Check if we should include this step
            if steps_to_plot is None or result['step'] in steps_to_plot:
                if metric_name in result:
                    values.append(result[metric_name])
                    metric_steps.append(result['step'])
        
        if values:
            custom_name = custom_names[j] if custom_names else None
            legend_name = get_legend_name(eval_folder, custom_name)
            color = colors[j % len(colors)]
            marker = markers[j % len(markers)]
            
            # Check if there's stderr data for this metric
            stderr_metric = metric_name + '_stderr'
            stderr_values = []
            for result in results:
                if steps_to_plot is None or result['step'] in steps_to_plot:
                    if metric_name in result and stderr_metric in result:
                        stderr_values.append(result[stderr_metric])
                    elif metric_name in result:
                        stderr_values.append(0)  # No stderr available for this step
            
            # Plot the main line
            plt.plot(metric_steps, values, marker=marker, markersize=6, 
                   color=color, label=legend_name, linewidth=2.5, alpha=0.9)
            
            # Plot error corridor if stderr data is available
            if stderr_values and any(stderr > 0 for stderr in stderr_values):
                lower_bounds = [v - s for v, s in zip(values, stderr_values)]
                upper_bounds = [v + s for v, s in zip(values, stderr_values)]
                plt.fill_between(metric_steps, lower_bounds, upper_bounds, 
                              color=color, alpha=0.2, linewidth=0)
    
    # Configure the plot
    plt.xlabel('Training Step (×1000)', fontsize=13, weight='bold')
    display_title = METRIC_TITLE_MAPPING.get(metric_name, metric_name)
    plt.ylabel(display_title, fontsize=13, weight='bold')
    plt.grid(True, alpha=0.2, linestyle='--', linewidth=0.5)
    
    # Set x-axis limits from 1000 to last datapoint with slight margins
    all_steps = []
    for results in all_results:
        for result in results:
            # Only include steps that match our filter and have the metric
            if (steps_to_plot is None or result['step'] in steps_to_plot) and metric_name in result:
                all_steps.append(result['step'])
    
    if all_steps:
        min_step = 1000
        max_step = max(all_steps)
        x_margin = (max_step - min_step) * 0.02  # 2% margin
        plt.xlim(min_step - x_margin, max_step + x_margin)
        # Set x-axis ticks to show simple integers (steps divided by 1000)
        unique_steps = sorted(set(all_steps))
        #unique_steps = [i for i in range(1000, 60000, 4000)]
        plt.xticks(unique_steps, [int(step/1000) for step in unique_steps])
    
    # Invert y-axis for ranking metrics (lower rank = better performance)
    if 'rank' in metric_name.lower():
        plt.gca().invert_yaxis()
        # Set y-axis limits from 1 to number of runs with slight margins
        y_margin = 0.1
        plt.ylim(len(eval_folders) + y_margin, 1 - y_margin)
    
    # Add legend if multiple runs
    if len(eval_folders) > 1:
        plt.legend(loc='upper left', frameon=True, fancybox=True, shadow=False, 
                  framealpha=0.9, edgecolor='gray', fontsize=11)
    
    # Add subtle background and improve spines
    ax = plt.gca()
    ax.set_facecolor('#fafafa')
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color('#333333')
    
    plt.tight_layout(pad=1.5)
    
    # Create assets folder if it doesn't exist
    assets_folder = '/fsx/luis_wiedmann/nanoVLM/plots_final'
    os.makedirs(assets_folder, exist_ok=True)
    
    # Generate filename for individual plot PDF
    if output_filename:
        pdf_file = os.path.join(assets_folder, f'{output_filename}_{metric_name}.pdf')
    elif len(eval_folders) == 1:
        folder_name = os.path.basename(eval_folders[0])
        pdf_file = os.path.join(assets_folder, f'{folder_name}_{metric_name}.pdf')
    else:
        pdf_file = os.path.join(assets_folder, f'comparison_{metric_name}.pdf')
    
    # Save as PDF with 300 DPI
    plt.savefig(pdf_file, format='pdf', dpi=300, bbox_inches='tight')
    print(f"Individual plot for '{metric_name}' saved to: {pdf_file}")
    
    # Also save as PNG
    png_file = pdf_file.replace('.pdf', '.png')
    plt.savefig(png_file, format='png', dpi=300, bbox_inches='tight')
    print(f"Individual plot for '{metric_name}' saved to: {png_file}")
    
    plt.close()

def save_csv_data(all_results, eval_folders, custom_names, metric_names, output_file, steps_to_plot=None):
    """Save the plot data to a CSV file."""
    # Prepare data for CSV
    csv_data = []
    
    for i, (results, eval_folder) in enumerate(zip(all_results, eval_folders)):
        # Get the run name
        custom_name = custom_names[i] if custom_names else None
        run_name = get_legend_name(eval_folder, custom_name)
        
        for result in results:
            step = result['step']
            # Only include steps that are plotted
            if steps_to_plot is None or step in steps_to_plot:
                for metric in metric_names:
                    if metric in result:
                        row_data = {
                            'run': run_name,
                            'step': step,
                            'metric': metric,
                            'value': result[metric]
                        }
                        
                        # Add stderr if available
                        stderr_metric = metric + '_stderr'
                        if stderr_metric in result:
                            row_data['stderr'] = result[stderr_metric]
                        
                        csv_data.append(row_data)
    
    # Convert to DataFrame and save
    if csv_data:
        df = pd.DataFrame(csv_data)
        # Generate CSV filename from plot filename
        csv_file = output_file.replace('.pdf', '.csv')
        df.to_csv(csv_file, index=False)
        print(f"Data saved to: {csv_file}")

def parse_args():
    """Parse command line arguments supporting both folder and folder:name format."""
    parser = argparse.ArgumentParser(
        description='Plot evaluation results from JSON files',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  python plot_eval_results.py /path/to/eval1
  python plot_eval_results.py Experiment1:/path/to/eval1 Experiment2:/path/to/eval2
  python plot_eval_results.py /path/to/eval1 --tasks vqa gqa
  python plot_eval_results.py Exp1:/path/to/eval1 Exp2:/path/to/eval2 --tasks mmlu"""
    )
    
    parser.add_argument('eval_folders', nargs='+',
                       help='Evaluation folder paths, optionally with custom names (folder:name)')
    parser.add_argument('--tasks', default=['docvqa_val_anls', 'infovqa_val_anls', 'mme_total_score', 'mmmu_val_mmmu_acc', 'mmstar_average', 'ocrbench_ocrbench_accuracy', 'scienceqa_exact_match', 'textvqa_val_exact_match', 'average'], nargs='+', #'ai2d_exact_match',
                       help='Specific tasks to plot (filters metrics containing these task names). Use "average_rank" for ranking-based summary metric.')
    parser.add_argument('--output', type=str,
                       help='Custom filename for the saved plot (without extension)')
    parser.add_argument('--steps', nargs='+', type=int,
                       help='Specific steps to plot (e.g., --steps 1000 2000 5000). If not specified, plots all available steps.')
    
    args = parser.parse_args()
    
    eval_folders = []
    custom_names = []
    
    for arg in args.eval_folders:
        if ':' in arg:
            name, folder = arg.rsplit(':', 1)
            eval_folders.append(folder)
            custom_names.append(name)
        else:
            eval_folders.append(arg)
            custom_names.append(None)
    
    # Check if all folders exist
    for eval_folder in eval_folders:
        if not os.path.exists(eval_folder):
            print(f"Error: Folder {eval_folder} does not exist")
            sys.exit(1)
    
    return eval_folders, custom_names, args.tasks, args.output, args.steps

def main():
    eval_folders, custom_names, tasks_to_plot, output_filename, steps_to_plot = parse_args()

    print("---------------------------")
    print(f"Plotting {output_filename}")
    
    # Load results from all folders
    all_results = []
    for eval_folder in eval_folders:
        print(f"Loading evaluation results from: {eval_folder}")
        results = load_eval_results(eval_folder, tasks_to_plot)
        if results:
            print(f"Found {len(results)} evaluation steps")
            
            # Check for missing evaluation steps if specific steps are requested
            if steps_to_plot:
                available_steps = {result['step'] for result in results}
                missing_steps = [step for step in steps_to_plot if step not in available_steps]
                
                if missing_steps:
                    folder_name = custom_names[eval_folders.index(eval_folder)] if custom_names and custom_names[eval_folders.index(eval_folder)] else os.path.basename(eval_folder)
                    print(f"Warning: {folder_name} missing evaluation steps: {missing_steps}")
            
            # Check for missing evaluations if specific tasks are requested
            if tasks_to_plot:
                available_metrics = set()
                for result in results:
                    available_metrics.update(k for k in result.keys() if k != 'step')
                
                missing_tasks = []
                for task in tasks_to_plot:
                    if task not in available_metrics and task not in ['average', 'average_rank']:
                        missing_tasks.append(task)
                
                if missing_tasks:
                    folder_name = custom_names[eval_folders.index(eval_folder)] if custom_names and custom_names[eval_folders.index(eval_folder)] else os.path.basename(eval_folder)
                    print(f"Warning: {folder_name} does not have evaluation for tasks: {missing_tasks}")
            
            all_results.append(results)
        else:
            print(f"No evaluation results found in {eval_folder}")
            all_results.append([])
    
    if any(all_results):
        # Compute ranking summary if requested
        if tasks_to_plot and 'average_rank' in tasks_to_plot:
            all_results = compute_ranking_summary(all_results, tasks_to_plot)
        
        plot_results(all_results, eval_folders, custom_names, tasks_to_plot, output_filename, steps_to_plot)
    else:
        print("No evaluation results found in any folder")

if __name__ == "__main__":
    main()