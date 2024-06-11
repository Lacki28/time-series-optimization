# cpu_list = ["./nb_results/", "./arima_results_(1, 0, 0)/", "./rf_results_cpu_16_200/", "lstm_results_cpu_final/",
#             "transformer_encoder_results_cpu_final/"]

cpu_list = ["./nb_results_mem/", "./arima_results_mem_(1, 0, 0)/", "./rf_results_mem_16_200/",
            "lstm_results_mem_final/", "transformer_encoder_results_mem_final/"]

name_list = ["NB", "ARIMA", "RF", "LSTM", "Trans."]

# Prepare the LaTeX table header
latex_table = r"""\begin{table}[htbp]
\centering
\caption{Mean squared error (MSE) prediction results of the methods for predicting the memory usage of the test data.\\ *Note: TS stands for Timestamps ahead, Mean is ($\bar{x}$) and Standard Deviation is ($\sigma$)}
\begin{center}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
& \multicolumn{2}{c|}{1 TS*} & \multicolumn{2}{c|}{2 TS}& \multicolumn{2}{c|}{3 TS} \\
\hline
Method& $\bar{x}$  & $\sigma$ & $\bar{x}$ & $\sigma$& $\bar{x}$ & $\sigma$ \\
\hline
"""

for index in range(5):
    file = cpu_list[index]
    method_name = name_list[index]
    # Read data from the file
    with open(file + "statistics_new/test_statistics.txt", "r") as file:
        data = file.readlines()
    latex_table += r"\textit{" + method_name + r"} & "
    non_empty_lines = [line.strip() for line in data if line.strip()]  # Filter out empty lines
    last_three_lines = non_empty_lines[:3]  # Get the last three non-empty lines
    concatenated_line = " ".join(last_three_lines)[:-2]
    print(concatenated_line)
    latex_table += concatenated_line + r" \\ " + "\n"

# Add the second part of the table header
latex_table += r"""\hline
\end{tabular}
\begin{tabular}{|c|c|c|c|c|c|c|}
\hline
\hline
& \multicolumn{2}{c|}{4 TS} & \multicolumn{2}{c|}{5 TS}& \multicolumn{2}{c|}{6 TS} \\
\hline
Method& $\bar{x}$  & $\sigma$ & $\bar{x}$ & $\sigma$& $\bar{x}$ & $\sigma$ \\
\hline
"""

for index in range(5):
    file = cpu_list[index]
    method_name = name_list[index]
    # Read data from the file
    with open(file + "statistics_new/test_statistics.txt", "r") as file:
        data = file.readlines()
    latex_table += r"\textit{" + method_name + r"} & "
    non_empty_lines = [line.strip() for line in data if line.strip()]  # Filter out empty lines
    last_three_lines = non_empty_lines[-3:]  # Get the last three non-empty lines
    concatenated_line = " ".join(last_three_lines)[:-2]
    print(concatenated_line)
    latex_table += concatenated_line + r" \\ " + "\n"
# Add the table footer
latex_table += r"""\hline
\end{tabular}
\end{center}
\label{cpuMSE}
\end{table}
"""

# Write the LaTeX code to a file
with open("output_table.txt", "w") as file:
    file.write(latex_table)
