# author: DSCI_522_Group_407
# date: 2020-01-22
#
"This script is used to download data (.csv file) from a
publicly accessible URL.

Usage: data_read.R --filepath=<filepath>

Example:
   Rscript data_read.R --filepath='https://opendatatc.blob.core.windows.net/opendatatc/NCDB_2017.csv'

Options:
<filepath>        Takes a file path (this is a required positional argument)
" -> doc

library(tidyverse)
library(docopt)
library(stringr)
library(testthat)

opt <- docopt(doc)

#' download data (.csv file) from a publicly accessible URL
#'
#' @param a character type of a filepath
#' @return a raw_data.csv file saved in the data folder
#' @examples
#' main('https://file.csv')
main <- function(file_path) {
    if (str_sub(file_path, start = -3) == 'csv') {
    download.file(url = opt$filepath, destfile = "data/raw_data.csv")
    data <- read.csv("data/raw_data.csv")
    } else {
    print("the url type shoule be .csv file")
    }
}

test_main <- function(){
  test_that("The file type should be .csv", {
      expect_match(
          main('https://opendatatc.blob.core.windows.net/opendatatc/NCDB_2017.html'), "the url type shoule be .csv file")
  })
}

test_main()

main(opt$filepath)
