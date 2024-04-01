#!/usr/bin/env python3
import os
import argparse
import sys


if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='Combines PDFs.')
  parser.add_argument('-i','--input_pdfs', nargs="+", required=True, help='Input pdf paths')
  parser.add_argument('-x','--number_x_plots', required=True, help='Number of x plots')
  parser.add_argument('-y','--number_y_plots', required=True, help='Number of y plots')
  parser.add_argument('-f', '--force_output', action="store_true", help='Make output even if output_pdf exists')
  parser.add_argument('-o','--output_pdf', required=True, help='Output pdf path')
  args = parser.parse_args()

  # https://github.com/rrthomas/pdfjam/releases/download/v3.10/pdfjam-3.10.tar.gz
  pdfjam_command = 'pdfjam'

  n_xplots = int(args.number_x_plots)
  n_yplots = int(args.number_y_plots)

  # Check if input_pdf paths exist and make all_pdfs string
  all_pdfs = ''
  for input_pdf in args.input_pdfs:
    if not os.path.isfile(input_pdf):
      print(f'[Error] Input {input_pdf} does not exist.')
      sys.exit()
    all_pdfs += f'{input_pdf} '

  # Check if output_pdf path exists
  if not args.force_output:
    if os.path.isfile(args.output_pdf):
      print(f'[Error] Output {args.output_pdf} exists.')
      sys.exit()

  ## 4 plots
  #input_pdfs = 'plots/true_z_mass__new_baseline__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/true_z_mass__new_baseline__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_phi0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_phi0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf'
  ## 6 plots
  #input_pdfs = 'plots/true_z_mass__new_baseline__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/true_z_mass__new_baseline__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_phi0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_phi0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/jet_eta0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_eta0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf'
  ## 9 plots
  #input_pdfs = 'plots/true_z_mass__new_baseline__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/true_z_mass__new_baseline__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_phi0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_phi0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/jet_eta0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_eta0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/jet_pt0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/jet_pt0__new_baseline_mass_window_njetge1__weight_fixxw_yearsxw_sigx20__lumi_nonorm_lin.pdf plots/ptt__new_baseline_mass_window__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf plots/ptt__new_baseline_mass_window__weight_fixxw_yearsxw_sigx20__shapes_lin.pdf'
  #output_pdf = 'test.pdf'

  ## Combine pdfs
  command = f'{pdfjam_command} {all_pdfs} --nup {n_xplots}x{n_yplots} --outfile {args.output_pdf}.1'
  print(command)
  os.system(command)

  # Find coordinates for trimming pdf
  pdf_width = 595.28
  pdf_height = 841.89
  plot_width = 567.
  plot_height = 544.
  x_factor = pdf_width/(n_xplots*plot_width)
  y_factor = pdf_height/(n_yplots*plot_height)
  min_factor = min(x_factor, y_factor)
  all_plots_width = min_factor * plot_width * n_xplots
  all_plots_height = min_factor * plot_height * n_yplots
  margin_width = (pdf_width - all_plots_width)
  margin_height = (pdf_height - all_plots_height)
  lower_x = margin_width / 2
  lower_y = margin_height / 2
  upper_x = lower_x + all_plots_width
  upper_y = lower_y + all_plots_height
  print(lower_x, lower_y, upper_x, upper_y)
  # Trim pdf
  command = f'gs -o {args.output_pdf} -sDEVICE=pdfwrite -c "[/CropBox [{lower_x} {lower_y} {upper_x} {upper_y}] /PAGES pdfmark" -f {args.output_pdf}.1;imgcat {args.output_pdf}'
  print(command)
  os.system(command)

  # Remove temp file
  command = f'rm -f {args.output_pdf}.1'
  print(command)
  os.system(command)
