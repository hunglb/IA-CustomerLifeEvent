# Sample Materials, provided under license.
# Licensed Materials - Property of IBM
# © Copyright IBM Corp. 2019. All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

# Load shap JavaScript
shapjs <- content(GET("https://github.com/slundberg/shap/raw/0849aa20551cf9825f9e294fcc29d7fbe7b9f932/shap/plots/resources/bundle.js"))

clientPanel <- function() {
  
  tabPanel(
    "Client View",
    value = "clientPanel",
    
    h1(textOutput("customerName")),
    br(),
    
    panel(
      br(),br(),
      fluidRow(
        column(3,
               div(id = "customerImage")),
        column(9,
               br(),
               tableOutput("customerTable")
        )
      ),
      br(),br(),
      fluidRow(
        column(3, 
               div(h3("Financial Profile:"), style="text-align:center;")
               ),
        column(9, tableOutput("customerFinancesTable"))
      ),
      br()
    ),
    
    fluidRow(
    column(7,
      panel(
        h3("Recent Customer Events"),
        br(),
        DTOutput("customerEventsTable")
      )
    ),
    column(5, 
      panel(
        h3("Event Distribution"),
        ggiraphOutput("customerEventsPlot", width = "600px", height = "400px")
      )
    )),
    
    panel(
      h3("Life Event Prediction "),
      br(),
      
      # Load shap JS
      tags$script(HTML(shapjs)),
      
      # Listen for responseInserted messages
      tags$script("
        Shiny.addCustomMessageHandler('responseInserted', function(elem) {
          var checkExist = setInterval(function() {
             if ($('#'+elem.id).length) {
                if (window.SHAP)
                  SHAP.ReactDom.render(
                    SHAP.React.createElement(SHAP.AdditiveForceVisualizer, elem.data),
                    document.getElementById(elem.id)
                  );
                clearInterval(checkExist);
             }
          }, 100); // check every 100ms
          
        });
      "),
      
      tags$div(
        id = "authPanel",
        column(4,
          panel(
            h4("Connect to ICP for Data API"),
            textInput("hostname", "ICP4D Hostname"),
            textInput("username", "ICP4D Username"),
            passwordInput("password", "ICP4D Password"),
            actionButton("authBtn", "Authenticate API", class = "btn-primary btn-lg btn-block", style = "max-width:300px", disabled = TRUE),
            tags$head(tags$style("#authError{color:red;}")),
            verbatimTextOutput("authError")
          ),
          style = "max-width:360px;"
        )
      ),
      hidden(
        tags$div(
          id = "deploymentPanel",
          column(4,
             panel(
               tags$h4("Model Scoring Pipeline Deployment"),
               pickerInput(
                 inputId = 'deploymentSelector',
                 label = 'Deployment Name:',
                 choices = list(),
                 options = pickerOptions(width = "auto", style = "btn-primary")
               ),
               tags$p(
                 tags$strong("Scoring URL: "),
                 textOutput(outputId = "scoring_url", inline = TRUE),
                 style = "word-wrap: break-word"
               ),
               tags$p(
                 tags$strong("Project Release: "),
                 textOutput(outputId = "release_name", inline = TRUE)
               ),
               tags$p(
                 tags$strong("Script Name: "),
                 textOutput(outputId = "script_name", inline = TRUE)
               ),
               tags$p(
                 tags$strong("Engine: "),
                 textOutput(outputId = "runtime", inline = TRUE)
               ))
          ),
          tags$div(id = "scoreBtnSection",
            column(4,
              br(),
              actionButton(
                 "scoreBtn",
                 "Predict Life Events",
                 class = "btn-primary btn-lg btn-block",
                 disabled = TRUE
               ),
              br(),
              h4("Input JSON:"),
              verbatimTextOutput("pipelineInput"),
              br(),
              tags$head(tags$style("#scoringError{color:red;}")),
              verbatimTextOutput("scoringError"))
          ),
          column(8,
             hidden(
               tags$div(id = "scoringResponse")
             )
          )
        )
      )
    )
  )
}

# Reactive server variables store (pervades across all sessions)
serverVariables = reactiveValues(deployments = list())

clientServer <- function(input, output, session, sessionVars) {
  
  observe({

    client <- clients[[toString(sessionVars$selectedClientId)]]

    # Update client name & image
    output$customerName <- renderText(client$name)
    removeUI(selector = "#customerImage > *")
    insertUI(
      selector = "#customerImage",
      where = "beforeEnd",
      ui = img(src = paste0("profiles/",client$image), style = "display: block;margin-left: auto;margin-right: auto;", width=150, height=150)
    )

    # Table displays for Customer View
    output$customerTable <- renderTable({
      demoDeets <- selection[,c("CUSTOMER_ID", "AGE_RANGE", "MARITAL_STATUS", "FAMILY_SIZE", "PROFESSION", "EDUCATION_LEVEL")]
      demoDeets[["CUSTOMER_ID"]] <- as.integer(demoDeets[["CUSTOMER_ID"]])
      demoDeets[["FAMILY_SIZE"]] <- as.integer(demoDeets[["FAMILY_SIZE"]])
      demoDeets[["ADDRESS"]] <- paste(selection[,"ADDRESS_HOME_CITY"], selection[,"ADDRESS_HOME_STATE"], sep = ', ')
      demoDeets[,c("CUSTOMER_ID", "AGE_RANGE", "ADDRESS", "MARITAL_STATUS", "FAMILY_SIZE", "PROFESSION", "EDUCATION_LEVEL")]
    }, bordered = TRUE, align = 'l')
    output$customerFinancesTable <- renderTable({
      finDeets <- selection[,c("ANNUAL_INCOME", "HOME_OWNER_INDICATOR", "MONTHLY_HOUSING_COST", "CREDIT_SCORE", "CREDIT_AUTHORITY_LEVEL")]
      finDeets[["ANNUAL_INCOME"]] <- dollar(finDeets[["ANNUAL_INCOME"]])
      finDeets[["MONTHLY_HOUSING_COST"]] <- dollar(finDeets[["MONTHLY_HOUSING_COST"]])
      finDeets
    }, bordered = TRUE, align = 'l')
    output$customerEventsTable <- renderDT(customerEvents[,c("EVENT_DATE", "NAME", "CATEGORY")], 
                                           rownames = FALSE, options = list(order = list(list(0, 'desc'))), style = 'bootstrap')

    # Load customer and event data for customer sessionVars$selectedClientId
    selection <- customer[customer$CUSTOMER_ID == sessionVars$selectedClientId,][1,]
    customerEvents <- events[events$CUSTOMER_ID == sessionVars$selectedClientId,]
    customerEvents <- merge(event_types, customerEvents, by="EVENT_TYPE_ID")
    customerEvents <- customerEvents[,c("CUSTOMER_ID", "EVENT_DATE", "NAME", "CATEGORY")]
    
    # Generate pie chart of event types distribution
    customerEventCounts <- data.frame(table(customerEvents$NAME)) %>%
      mutate(
        percentage = Freq / sum(Freq),
        hover_text = paste0(Var1, ": ", round(100 * percentage, 1), "%")
      )
    count_plot <- ggplot(customerEventCounts, aes(y = Freq, fill = Var1)) +
      geom_bar_interactive(
        aes(x = 1, tooltip = hover_text),
        stat = "identity",
        show.legend = TRUE
      ) + 
      coord_polar(theta = "y") +
      theme_void() +
      theme(legend.title=element_text(size=16), 
            legend.text=element_text(size=12))
    count_plot <- count_plot + guides(fill=guide_legend(title="Event Types"))
    output$customerEventsPlot <- renderggiraph(ggiraph(ggobj = count_plot, width_svg=6, height_svg=4))

    # Reset scoring
    removeUI(selector = "#scoringResponse > *", multiple = TRUE)
    shinyjs::hide(id = "scoringResponse")
    shinyjs::show(id = "scoreBtnSection")
    output$scoringError <- renderText('')
    sessionVars$pipelineInput <- list(dataset_name = 'event.csv', cust_id = sessionVars$selectedClientId, sc_end_date = '2017-08-01')
    output$pipelineInput <- renderText(toJSON(sessionVars$pipelineInput, indent = 2))
  })
  
  # Set default hostname for ICP4D API
  observeEvent(session$clientData$url_hostname, {
    updateTextInput(session, "hostname", value = session$clientData$url_hostname)
  })
  
  # Enable buttons when inputs are provided
  observe({
    toggleState("authBtn", nchar(input$hostname) > 0 && nchar(input$username) > 0 && nchar(input$password) > 0)
    toggleState("scoreBtn", nchar(input$endpoint) > 0 && nchar(input$token) > 0 && length(input$allCustomers_rows_selected) > 0)
  })
  
  # Handle ICP4D API authentication button
  observeEvent(input$authBtn, {
    shinyjs::disable("authBtn")
    tryCatch({
      serverVariables$deployments <- collectDeployments(input$hostname, input$username, input$password, "LFE_Scoring_Pipeline.py")
    }, warning = function(w) {
      output$authError <- renderText(w$message)
    }, error = function(e) {
      output$authError <- renderText(e$message)
    })
    shinyjs::enable("authBtn")
  })
  
  observe({
    if(length(serverVariables$deployments) > 0) {
      updateSelectInput(session, "deploymentSelector", choices = names(serverVariables$deployments))
      shinyjs::hide(id = "authPanel")
      shinyjs::show(id = "deploymentPanel")
    }
  })
  
  # Handle model deployment dropdown switching
  observeEvent(input$deploymentSelector, {
    selectedDeployment <- serverVariables$deployments[[input$deploymentSelector]]
    output$release_name <- renderText(selectedDeployment$release_name)
    output$scoring_url <- renderText(selectedDeployment$scoring_url)
    output$script_name <- renderText(selectedDeployment$deployment_asset_name)
    output$runtime <- renderText(selectedDeployment$runtime_definition_name)
    toggleState("scoreBtn", nchar(selectedDeployment$deployment_url) > 0 && nchar(selectedDeployment$deployment_token) > 0)
  })
  
  # Handle model deployment scoring button
  observeEvent(input$scoreBtn, {
    shinyjs::disable("scoreBtn")
    selectedDeployment <- serverVariables$deployments[[input$deploymentSelector]]
    
    response <- scoreModelDeployment(selectedDeployment$scoring_url, selectedDeployment$deployment_token, sessionVars$pipelineInput)
    
    if(length(response$error) > 0) {
      output$scoringError <- renderText(toString(response$error))
    }
    else if(length(response$result) > 0) {
      shinyjs::hide(id = "scoreBtnSection")
      shinyjs::show(id = "scoringResponse")
      
      for(event_type in names(response$result)) {
        event_result <- response$result[[event_type]]
        
        insertUI(
          selector = "#scoringResponse",
          where = "beforeEnd",
          ui = panel(
            h3(paste("Life Event:", substr(event_type, 5, nchar(event_type)))),
            fluidRow(
              column(6,
                     ggiraphOutput(paste0("probPlot-",event_type), width = "600px", height = "400px")
              ),
              column(6,
                     p(
                       tableOutput(paste0("explain-table-",event_type))
                     )
              )
            ),
            p(
              strong("Explanation Plot: ")
            ),
            HTML(event_result$explain_plot_html)
          )
        )
        
        session$sendCustomMessage('responseInserted', 
                                  list(
                                    id=event_result$explain_plot_elem_id,
                                    data=fromJSON(event_result$explain_plot_data))
                                  )
      }
      
      lapply(names(response$result), function(event_type) {
        
        event_result <- response$result[[event_type]]
        
        # render high impact features table
        vertical <- t(data.frame(event_result$explain))
        Impact <- vertical[order(vertical, decreasing = TRUE),]
        df <- data.frame(Impact)
        dispTable <- tibble::rownames_to_column(df, "Highest Impact Features")
        output[[paste0("explain-table-",event_type)]] <- renderTable(dispTable, bordered = TRUE)
        
        # generate probability pie
        probDF <- data.frame(event_result$probabilities)
        colnames(probDF) <- "Probability"
        row.names(probDF) <- c("FALSE", "TRUE")
        probDF <- tibble::rownames_to_column(probDF, "Prediction")
        probDF <- probDF %>%
          mutate(percentage = paste0(round(100 * Probability, 1), "%")) %>%
          mutate(hover_text = paste0(Prediction, ": ", percentage))
        
        probPlot <- ggplot(probDF, aes(y = Probability, fill = Prediction)) +
          geom_bar_interactive(
            aes(x = 1, tooltip = hover_text),
            width = 0.4,
            stat = "identity",
            show.legend = TRUE
          ) + 
          annotate("text", x = 0, y = 0, size = 12,
                   label = probDF[["percentage"]][probDF[["Prediction"]] == "TRUE"]
                  ) +
          coord_polar(theta = "y") +
          theme_void() +
          theme(legend.title=element_text(size=22),
                legend.text=element_text(size=16)) +
          guides(fill = guide_legend(reverse=TRUE))
        
        output[[paste0("probPlot-",event_type)]] <- renderggiraph(ggiraph(ggobj = probPlot, width_svg=6, height_svg=4))
      })
      
    } else {
      output$scoringError <- renderText(response)
    }
    shinyjs::enable("scoreBtn")
  })
}
