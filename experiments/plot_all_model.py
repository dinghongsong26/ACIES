"""IEEE-style grouped bar charts of latency and cost per model, with vs without lessons.

Earlier iterations of the figure are kept commented out below; the active version is at the
bottom of the file.
"""

# # # # # # import matplotlib.pyplot as plt
# # # # # # import numpy as np
# # # # # # import os

# # # # # # # 1. Prepare Data
# # # # # # models = ['Claude-Opus-4-7', 'GPT-5.5', 'Gemini-3.1-Pro', 
# # # # # #           'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-k2.6']

# # # # # # time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
# # # # # # time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

# # # # # # cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
# # # # # # cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

# # # # # # # 2. Style Settings (IEEE Standard)
# # # # # # plt.rcParams['font.family'] = 'serif'
# # # # # # plt.rcParams['font.serif'] = ['Times New Roman']
# # # # # # plt.rcParams['axes.labelsize'] = 10
# # # # # # plt.rcParams['xtick.labelsize'] = 9
# # # # # # plt.rcParams['ytick.labelsize'] = 9
# # # # # # plt.rcParams['legend.fontsize'] = 8

# # # # # # def plot_ieee_perfect_centered(data_wo, data_w, ylabel, filename):
# # # # # #     fig, ax = plt.subplots(figsize=(4.8, 3.2)) 
    
# # # # # #     x = np.arange(len(models))
# # # # # #     width = 0.36 
    
# # # # # #     # Matching colors from image_afb3f8.png
# # # # # #     color_wo = 'white'
# # # # # #     color_w = '#4682B4' 
    
# # # # # #     ax.bar(x - width/2, data_wo, width, label='w/o Lessons', 
# # # # # #            color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
# # # # # #     ax.bar(x + width/2, data_w, width, label='with Lessons', 
# # # # # #            color=color_w, edgecolor='black', linewidth=0.8)

# # # # # #     ax.set_ylabel(ylabel, fontweight='bold')
# # # # # #     ax.set_xticks(x)
    
# # # # # #     # --- KEY ADJUSTMENT FOR CENTERING & SLANTING ---
# # # # # #     # rotation=20: Slight slant for readability
# # # # # #     # ha='center': Keeps the text centered relative to the bar group
# # # # # #     # va='top': Pushes the text below the axis line to prevent crossing
# # # # # #     # rotation_mode='anchor': Ensures rotation happens around the 'center' alignment point
# # # # # #     ax.set_xticklabels(models, rotation=12, ha='center', va='top', rotation_mode='anchor') 

# # # # # #     # Prevent legend overlap (45% buffer)
# # # # # #     max_val = max(max(data_wo), max(data_w))
# # # # # #     ax.set_ylim(0, max_val * 1.45) 

# # # # # #     ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False, framealpha=1)
# # # # # #     ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
# # # # # #     # Fine-tune layout for IEEE double-column papers
# # # # # #     plt.tight_layout()
    
# # # # # #     save_path = os.path.join(os.getcwd(), filename)
# # # # # #     plt.savefig(save_path, dpi=600, bbox_inches='tight')
# # # # # #     plt.show()

# # # # # # # 3. Save Figures
# # # # # # plot_ieee_perfect_centered(time_wo, time_w, 'Latency (s)', 'latency_ieee_centered.pdf')
# # # # # # plot_ieee_perfect_centered(cost_wo, cost_w, 'Total Cost (USD)', 'cost_ieee_centered.pdf')


# # # # # import matplotlib.pyplot as plt
# # # # # import numpy as np
# # # # # import os


# # # # # models = ['Claude-Opus-4.7', 'GPT-5.5', 'Gemini-3.1-Pro', 
# # # # #           'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-K2.6']

# # # # # # Latency data (e2e time)
# # # # # time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
# # # # # time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

# # # # # # Cost data
# # # # # total_cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
# # # # # total_cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

# # # # # translator_cost_wo = [0.062, 0.065, 0.024, 0.004, 0.014, 0.042]
# # # # # translator_cost_w = [0.061, 0.062, 0.024, 0.005, 0.010, 0.043]

# # # # # decomposer_cost_wo = [0.149, 0.256, 0.051, 0.017, 0.035, 0.122]
# # # # # decomposer_cost_w = [0.079, 0.151, 0.026, 0.007, 0.020, 0.043]

# # # # # # 2. Global Style Settings (IEEE Standards)
# # # # # plt.rcParams['font.family'] = 'serif'
# # # # # plt.rcParams['font.serif'] = ['Times New Roman']
# # # # # plt.rcParams['axes.labelsize'] = 10
# # # # # plt.rcParams['xtick.labelsize'] = 9
# # # # # plt.rcParams['ytick.labelsize'] = 9
# # # # # plt.rcParams['legend.fontsize'] = 8

# # # # # def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):
# # # # #     """
# # # # #     Generates a standardized bar chart suitable for double-column IEEE papers.
# # # # #     """
# # # # #     fig, ax = plt.subplots(figsize=(4.8, 3.2)) 
    
# # # # #     x = np.arange(len(models))
# # # # #     width = 0.36 
    
# # # # #     # Colors matching the reference style
# # # # #     color_wo = 'white'
# # # # #     color_w = '#4682B4' # SteelBlue
    
# # # # #     # Plot bars
# # # # #     ax.bar(x - width/2, data_wo, width, label='w/o Lessons', 
# # # # #            color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
# # # # #     ax.bar(x + width/2, data_w, width, label='with Lessons', 
# # # # #            color=color_w, edgecolor='black', linewidth=0.8)

# # # # #     # Labeling
# # # # #     ax.set_ylabel(ylabel, fontweight='bold')
# # # # #     ax.set_xticks(x)
    
# # # # #     # Set tick labels with a slight rotation for better fitting
# # # # #     ax.set_xticklabels(models, rotation=12, ha='center', va='top', rotation_mode='anchor') 

# # # # #     # Adjust Y-limit to leave room for legend (45% buffer)
# # # # #     max_val = max(max(data_wo), max(data_w))
# # # # #     ax.set_ylim(0, max_val * 1.45) 

# # # # #     # Legend and Grid configuration
# # # # #     ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False, framealpha=1)
# # # # #     ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
# # # # #     # Fine-tune layout to prevent clipping
# # # # #     plt.tight_layout()
    
# # # # #     # Save as high-resolution PDF/PNG
# # # # #     save_path = os.path.join(os.getcwd(), filename)
# # # # #     plt.savefig(save_path, dpi=600, bbox_inches='tight')
# # # # #     print(f"Successfully saved: {filename}")
# # # # #     plt.close()

# # # # # # 3. Execution: Generate the 4 Figures
# # # # # if __name__ == "__main__":
# # # # #     # Figure 1: Latency (End-to-End Time)
# # # # #     plot_ieee_bar_chart(time_wo, time_w, 'Latency (s)', 'fig_latency.pdf')
    
# # # # #     # Figure 2: Total Cost
# # # # #     plot_ieee_bar_chart(total_cost_wo, total_cost_w, 'Total Cost (USD)', 'fig_total_cost.pdf')
    
# # # # #     # Figure 3: Translator Cost
# # # # #     plot_ieee_bar_chart(translator_cost_wo, translator_cost_w, 'Translator Cost (USD)', 'fig_translator_cost.pdf')
    
# # # # #     # Figure 4: Decomposer Cost
# # # # #     plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w, 'Decomposer Cost (USD)', 'fig_decomposer_cost.pdf')



# # # # import matplotlib.pyplot as plt
# # # # import numpy as np
# # # # import os


# # # # models = ['Claude-Opus-4.7', 'GPT-5.5', 'Gemini-3.1-Pro', 
# # # #           'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-K2.6']

# # # # # Latency data (e2e time)
# # # # time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
# # # # time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

# # # # # Cost data
# # # # total_cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
# # # # total_cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

# # # # translator_cost_wo = [0.062, 0.065, 0.024, 0.004, 0.014, 0.042]
# # # # translator_cost_w = [0.061, 0.062, 0.024, 0.005, 0.010, 0.043]

# # # # decomposer_cost_wo = [0.149, 0.256, 0.051, 0.017, 0.035, 0.122]
# # # # decomposer_cost_w = [0.079, 0.151, 0.026, 0.007, 0.020, 0.043]

# # # # # -----------------------------
# # # # # ONLY CHANGE: Larger Fonts
# # # # # -----------------------------
# # # # plt.rcParams['font.family'] = 'serif'
# # # # plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

# # # # plt.rcParams['axes.labelsize'] = 15
# # # # plt.rcParams['xtick.labelsize'] = 13
# # # # plt.rcParams['ytick.labelsize'] = 13
# # # # plt.rcParams['legend.fontsize'] = 12

# # # # def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):
# # # #     """
# # # #     Generates a standardized bar chart suitable for double-column IEEE papers.
# # # #     """
# # # #     fig, ax = plt.subplots(figsize=(4.8, 3.2)) 
    
# # # #     x = np.arange(len(models))
# # # #     width = 0.36 
    
# # # #     # Colors matching the reference style
# # # #     color_wo = 'white'
# # # #     color_w = '#4682B4' # SteelBlue
    
# # # #     # Plot bars
# # # #     ax.bar(x - width/2, data_wo, width, label='w/o Lessons', 
# # # #            color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
# # # #     ax.bar(x + width/2, data_w, width, label='with Lessons', 
# # # #            color=color_w, edgecolor='black', linewidth=0.8)

# # # #     # Labeling
# # # #     ax.set_ylabel(ylabel, fontweight='bold')
# # # #     ax.set_xticks(x)
    
# # # #     # Keep your original alignment settings
# # # #     ax.set_xticklabels(models, rotation=12, ha='center', va='top', rotation_mode='anchor') 

# # # #     # Adjust Y-limit to leave room for legend (45% buffer)
# # # #     max_val = max(max(data_wo), max(data_w))
# # # #     ax.set_ylim(0, max_val * 1.45) 

# # # #     # Legend and Grid configuration
# # # #     ax.legend(loc='upper left', frameon=True, edgecolor='black', fancybox=False, framealpha=1)
# # # #     ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    
# # # #     # Fine-tune layout to prevent clipping
# # # #     plt.tight_layout()
    
# # # #     # Save as high-resolution PDF/PNG
# # # #     save_path = os.path.join(os.getcwd(), filename)
# # # #     plt.savefig(save_path, dpi=600, bbox_inches='tight')
# # # #     print(f"Successfully saved: {filename}")
# # # #     plt.close()

# # # # # -----------------------------
# # # # # Execution
# # # # # -----------------------------
# # # # if __name__ == "__main__":
# # # #     plot_ieee_bar_chart(time_wo, time_w, 'Latency (s)', 'fig_latency.pdf')
# # # #     plot_ieee_bar_chart(total_cost_wo, total_cost_w, 'Total Cost (USD)', 'fig_total_cost.pdf')
# # # #     plot_ieee_bar_chart(translator_cost_wo, translator_cost_w, 'Translator Cost (USD)', 'fig_translator_cost.pdf')
# # # #     plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w, 'Decomposer Cost (USD)', 'fig_decomposer_cost.pdf')


# # # import matplotlib.pyplot as plt
# # # import numpy as np
# # # import os


# # # models = ['Claude-Opus-4.7', 'GPT-5.5', 'Gemini-3.1-Pro', 
# # #           'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-K2.6']

# # # # Latency data (e2e time)
# # # time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
# # # time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

# # # # Cost data
# # # total_cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
# # # total_cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

# # # translator_cost_wo = [0.062, 0.065, 0.024, 0.004, 0.014, 0.042]
# # # translator_cost_w = [0.061, 0.062, 0.024, 0.005, 0.010, 0.043]

# # # decomposer_cost_wo = [0.149, 0.256, 0.051, 0.017, 0.035, 0.122]
# # # decomposer_cost_w = [0.079, 0.151, 0.026, 0.007, 0.020, 0.043]

# # # # -----------------------------
# # # # Larger Fonts (keep your previous upgrade)
# # # # -----------------------------
# # # plt.rcParams['font.family'] = 'serif'
# # # plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

# # # plt.rcParams['axes.labelsize'] = 15
# # # plt.rcParams['xtick.labelsize'] = 13
# # # plt.rcParams['ytick.labelsize'] = 13
# # # plt.rcParams['legend.fontsize'] = 12

# # # def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):
# # #     """
# # #     Wider layout + lower height for better x-axis readability.
# # #     """

# # #     # ✅ KEY CHANGE 1: Make figure wider and shorter
# # #     fig, ax = plt.subplots(figsize=(6.8, 2.8))  

# # #     x = np.arange(len(models))
# # #     width = 0.36 
    
# # #     color_wo = 'white'
# # #     color_w = '#4682B4'
    
# # #     ax.bar(x - width/2, data_wo, width, label='w/o Lessons', 
# # #            color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
# # #     ax.bar(x + width/2, data_w, width, label='with Lessons', 
# # #            color=color_w, edgecolor='black', linewidth=0.8)

# # #     ax.set_ylabel(ylabel, fontweight='bold')
# # #     ax.set_xticks(x)

# # #     ax.set_xticklabels(models, rotation=12, ha='center', va='top', rotation_mode='anchor') 

# # #     # Slightly reduce headroom (since legend moved)
# # #     max_val = max(max(data_wo), max(data_w))
# # #     ax.set_ylim(0, max_val * 1.25)

# # #     # ✅ KEY CHANGE 2: Move legend ABOVE the plot (no overlap)
# # #     ax.legend(
# # #         loc='lower center',
# # #         bbox_to_anchor=(0.5, 1.02),
# # #         ncol=2,
# # #         frameon=True,
# # #         edgecolor='black',
# # #         fancybox=False,
# # #         framealpha=1
# # #     )

# # #     ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

# # #     plt.tight_layout()

# # #     save_path = os.path.join(os.getcwd(), filename)
# # #     plt.savefig(save_path, dpi=600, bbox_inches='tight')
# # #     print(f"Successfully saved: {filename}")
# # #     plt.close()

# # # # -----------------------------
# # # # Execution
# # # # -----------------------------
# # # if __name__ == "__main__":
# # #     plot_ieee_bar_chart(time_wo, time_w, 'Latency (s)', 'fig_latency.pdf')
# # #     plot_ieee_bar_chart(total_cost_wo, total_cost_w, 'Total Cost (USD)', 'fig_total_cost.pdf')
# # #     plot_ieee_bar_chart(translator_cost_wo, translator_cost_w, 'Translator Cost (USD)', 'fig_translator_cost.pdf')
# # #     plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w, 'Decomposer Cost (USD)', 'fig_decomposer_cost.pdf')


# # import matplotlib.pyplot as plt
# # import numpy as np
# # import os
# # from matplotlib.ticker import AutoMinorLocator

# # models = ['Claude-Opus-4.7', 'GPT-5.5', 'Gemini-3.1-Pro', 
# #           'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-K2.6']

# # # Latency data (e2e time)
# # time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
# # time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

# # # Cost data
# # total_cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
# # total_cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

# # translator_cost_wo = [0.062, 0.065, 0.024, 0.004, 0.014, 0.042]
# # translator_cost_w = [0.061, 0.062, 0.024, 0.005, 0.010, 0.043]

# # decomposer_cost_wo = [0.149, 0.256, 0.051, 0.017, 0.035, 0.122]
# # decomposer_cost_w = [0.079, 0.151, 0.026, 0.007, 0.020, 0.043]

# # # -----------------------------
# # # Larger Fonts (keep)
# # # -----------------------------
# # plt.rcParams['font.family'] = 'serif'
# # plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

# # plt.rcParams['axes.labelsize'] = 15
# # plt.rcParams['xtick.labelsize'] = 13
# # plt.rcParams['ytick.labelsize'] = 13
# # plt.rcParams['legend.fontsize'] = 12

# # def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):
# #     """
# #     Wider layout + in-plot legend + finer y-axis ticks
# #     """

# #     # ✅ Slightly wider than before
# #     fig, ax = plt.subplots(figsize=(7.6, 2.8))  

# #     x = np.arange(len(models))
# #     width = 0.36 
    
# #     color_wo = 'white'
# #     color_w = '#4682B4'
    
# #     ax.bar(x - width/2, data_wo, width, label='w/o Lessons', 
# #            color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
# #     ax.bar(x + width/2, data_w, width, label='with Lessons', 
# #            color=color_w, edgecolor='black', linewidth=0.8)

# #     ax.set_ylabel(ylabel, fontweight='bold')
# #     ax.set_xticks(x)

# #     ax.set_xticklabels(models, rotation=12, ha='center', va='top', rotation_mode='anchor') 

# #     # Slightly reduced headroom
# #     max_val = max(max(data_wo), max(data_w))
# #     ax.set_ylim(0, max_val * 1.2)

# #     # ✅ Legend INSIDE but safely placed (top-left empty region)
# #     ax.legend(
# #         loc='upper left',
# #         frameon=True,
# #         edgecolor='black',
# #         fancybox=False,
# #         framealpha=1
# #     )

# #     # ✅ Finer Y-axis ticks
# #     ax.yaxis.set_minor_locator(AutoMinorLocator(2))  # add minor ticks

# #     # Major grid
# #     ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)

# #     # Minor grid (lighter)
# #     ax.grid(axis='y', which='minor', linestyle=':', alpha=0.25, linewidth=0.4)

# #     plt.tight_layout()

# #     save_path = os.path.join(os.getcwd(), filename)
# #     plt.savefig(save_path, dpi=600, bbox_inches='tight')
# #     print(f"Successfully saved: {filename}")
# #     plt.close()

# # # -----------------------------
# # # Execution
# # # -----------------------------
# # if __name__ == "__main__":
# #     plot_ieee_bar_chart(time_wo, time_w, 'Latency (s)', 'fig_latency.pdf')
# #     plot_ieee_bar_chart(total_cost_wo, total_cost_w, 'Total Cost (USD)', 'fig_total_cost.pdf')
# #     plot_ieee_bar_chart(translator_cost_wo, translator_cost_w, 'Translator Cost (USD)', 'fig_translator_cost.pdf')
# #     plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w, 'Decomposer Cost (USD)', 'fig_decomposer_cost.pdf')



# import matplotlib.pyplot as plt
# import numpy as np
# import os
# from matplotlib.ticker import AutoMinorLocator

# models = ['Claude-Opus-4.7', 'GPT-5.5', 'Gemini-3.1-Pro', 
#           'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-K2.6']

# time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
# time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

# total_cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
# total_cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

# translator_cost_wo = [0.062, 0.065, 0.024, 0.004, 0.014, 0.042]
# translator_cost_w = [0.061, 0.062, 0.024, 0.005, 0.010, 0.043]

# decomposer_cost_wo = [0.149, 0.256, 0.051, 0.017, 0.035, 0.122]
# decomposer_cost_w = [0.079, 0.151, 0.026, 0.007, 0.020, 0.043]

# # -----------------------------
# # Style
# # -----------------------------
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

# plt.rcParams['axes.labelsize'] = 15
# plt.rcParams['xtick.labelsize'] = 13
# plt.rcParams['ytick.labelsize'] = 13
# plt.rcParams['legend.fontsize'] = 12

# def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):

#     fig, ax = plt.subplots(figsize=(7.6, 2.8))

#     x = np.arange(len(models))
#     width = 0.36

#     color_wo = 'white'
#     color_w = '#4682B4'

#     ax.bar(x - width/2, data_wo, width, label='w/o Lessons',
#            color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
#     ax.bar(x + width/2, data_w, width, label='with Lessons',
#            color=color_w, edgecolor='black', linewidth=0.8)

#     ax.set_ylabel(ylabel, fontweight='bold')
#     ax.set_xticks(x)

#     ax.set_xticklabels(models, rotation=12, ha='center', va='top', rotation_mode='anchor')

#     max_val = max(max(data_wo), max(data_w))
#     ax.set_ylim(0, max_val * 1.2)

#     # -----------------------------
#     # Legend position switch logic
#     # -----------------------------
#     if 'Latency' in ylabel:
#         legend_loc = 'upper left'
#     else:
#         legend_loc = 'upper right'

#     ax.legend(
#         loc=legend_loc,
#         frameon=True,
#         edgecolor='black',
#         fancybox=False,
#         framealpha=1
#     )

#     # Fine y-axis ticks
#     ax.yaxis.set_minor_locator(AutoMinorLocator(2))
#     ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
#     ax.grid(axis='y', which='minor', linestyle=':', alpha=0.25, linewidth=0.4)

#     plt.tight_layout()

#     save_path = os.path.join(os.getcwd(), filename)
#     plt.savefig(save_path, dpi=600, bbox_inches='tight')
#     print(f"Saved: {filename}")
#     plt.close()

# # -----------------------------
# # Run
# # -----------------------------
# if __name__ == "__main__":
#     plot_ieee_bar_chart(time_wo, time_w, 'Latency (s)', 'fig_latency.pdf')
#     plot_ieee_bar_chart(total_cost_wo, total_cost_w, 'Total Cost (USD)', 'fig_total_cost.pdf')
#     plot_ieee_bar_chart(translator_cost_wo, translator_cost_w, 'Translator Cost (USD)', 'fig_translator_cost.pdf')
#     plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w, 'Decomposer Cost (USD)', 'fig_decomposer_cost.pdf')


import matplotlib.pyplot as plt
import numpy as np
import os
from matplotlib.ticker import AutoMinorLocator

models = ['Claude-Opus-4.7', 'GPT-5.5', 'Gemini-3.1-Pro', 
          'DeepSeek-V4-Pro', 'GLM-5.1', 'Kimi-K2.6']

time_wo = [35.5, 126.7, 158.5, 530.2, 196.3, 800.4]
time_w = [27.7, 78.3, 59.6, 331.7, 101.9, 651.9]

total_cost_wo = [0.211, 0.322, 0.076, 0.022, 0.049, 0.164]
total_cost_w = [0.141, 0.213, 0.052, 0.013, 0.040, 0.106]

translator_cost_wo = [0.062, 0.065, 0.024, 0.004, 0.014, 0.042]
translator_cost_w = [0.061, 0.062, 0.024, 0.005, 0.010, 0.043]

decomposer_cost_wo = [0.149, 0.256, 0.051, 0.017, 0.035, 0.122]
decomposer_cost_w = [0.079, 0.151, 0.026, 0.007, 0.020, 0.043]

# -----------------------------
# Style (bigger fonts)
# -----------------------------
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman', 'Times']

plt.rcParams['axes.labelsize'] = 17
plt.rcParams['xtick.labelsize'] = 15
plt.rcParams['ytick.labelsize'] = 15
plt.rcParams['legend.fontsize'] = 13

def plot_ieee_bar_chart(data_wo, data_w, ylabel, filename):
    """Save a paired (w/o vs with lessons) bar chart over models to `filename`."""

    fig, ax = plt.subplots(figsize=(7.6, 2.8))

    x = np.arange(len(models))

    # ✅ thinner bars
    width = 0.32

    color_wo = 'white'
    color_w = '#4682B4'

    ax.bar(x - width/2, data_wo, width, label='w/o Lessons',
           color=color_wo, edgecolor='black', hatch='////', linewidth=0.8)
    ax.bar(x + width/2, data_w, width, label='with Lessons',
           color=color_w, edgecolor='black', linewidth=0.8)

    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_xticks(x)

    ax.set_xticklabels(models, rotation=8, ha='center', va='top', rotation_mode='anchor')

    max_val = max(max(data_wo), max(data_w))
    ax.set_ylim(0, max_val * 1.2)

    # legend position logic
    if 'Latency' in ylabel:
        legend_loc = 'upper left'
    else:
        legend_loc = 'upper right'

    ax.legend(
        loc=legend_loc,
        frameon=True,
        edgecolor='black',
        fancybox=False,
        framealpha=1
    )

    # finer y-axis
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))
    ax.grid(axis='y', linestyle='--', alpha=0.4, linewidth=0.5)
    ax.grid(axis='y', which='minor', linestyle=':', alpha=0.25, linewidth=0.4)

    plt.tight_layout()

    save_path = os.path.join(os.getcwd(), filename)
    plt.savefig(save_path, dpi=600, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# -----------------------------
# Run
# -----------------------------
if __name__ == "__main__":
    plot_ieee_bar_chart(time_wo, time_w, 'E2E Latency (s)', 'fig_latency.pdf')
    plot_ieee_bar_chart(total_cost_wo, total_cost_w, 'Total Cost ($)', 'fig_total_cost.pdf')
    plot_ieee_bar_chart(translator_cost_wo, translator_cost_w, 'Translator Cost ($)', 'fig_translator_cost.pdf')
    plot_ieee_bar_chart(decomposer_cost_wo, decomposer_cost_w, 'Decomposer Cost ($)', 'fig_decomposer_cost.pdf')