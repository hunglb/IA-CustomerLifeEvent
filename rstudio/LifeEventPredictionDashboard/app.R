# Title     : Life Event Prediction Dashboard
# Objective : Demonstrate Life Event Prediction Industry Accelerator for Data Science Use Cases

# Sample Materials, provided under license.
# Licensed Materials - Property of IBM
# © Copyright IBM Corp. 2019. All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

### Install any missing packages ###

# Determine packages to install among requirements
list.of.packages <- c("shinyWidgets", "shinyjs", "ggplot2", "ggiraph", "dplyr")
new.packages <- list.of.packages[!(list.of.packages %in% installed.packages()[,"Package"])]

if(length(new.packages)) {  # check if there's anything to install
  
  # set default libpath
  if (Sys.getenv("DSX_PROJECT_DIR")!=""){                           # If we are in WSL,
    target <- paste0(Sys.getenv("DSX_PROJECT_DIR"),"/packages/R")   # default to project packages/R
  } else {                                                          # Otherwise,
    target <- .libPaths()[1]                                        # default to first libPath (default)
  }
  
  # check for other valid libpaths
  for(libpath in .libPaths()) {           # check the .libPaths
    if(file.access(libpath, 2) == 0) {    # if we have write access to a libpath, use it
      target <- libpath
      break
    }
  }
  
  # Install the packages
  print(paste("Installing ", paste(new.packages, collapse = ", "), "to", target))
  install.packages(new.packages, lib = target)
}

# Load required packages
library(shiny)
library(shinyWidgets)
library(shinyjs)
library(DT)
library(httr)
library(rjson)
library(ggplot2)
library(ggiraph)
library(dplyr)

# Load datasets & ICP4D API functions
source("lib/load-data.R")
source("lib/icp4d-api.R")

# Load panels
source("homePanel.R")
source("clientPanel.R")

# Serve customer profile images
addResourcePath('profiles', normalizePath('../../misc/rshiny/profiles'))

ui <- navbarPage(
  "Life Event Prediction",
  id = "lfeNav",
  
  homePanel(),
  
  clientPanel()
)

server <- function(input, output, session) {
  
  sessionVars <- reactiveValues(selectedClientId = 1039)
  
  homeServer(input, output, session, sessionVars)
  
  clientServer(input, output, session, sessionVars)
}

# Create Shiny app
shinyApp(ui, server)
