# Sample Materials, provided under license.
# Licensed Materials - Property of IBM
# © Copyright IBM Corp. 2019. All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

clientButton <- function(id, name, income, image) {
  tags$p(
    actionButton(paste0('client-btn-', id),
                 fluidRow(
                   column(3, 
                          tags$img(src = image, width = "100px", height = "100px")
                   ),
                   column(9,
                          tags$h3(name),
                          tags$h4(paste('Annual Income: ', income))
                   )
                 ),
                 style="width:100%"
    )
  )
}

homePanel <- function() {
  
  tabPanel(
    "Dashboard",
    tags$head(
      tags$style(HTML("
        .datatables {
          width: 100% !important;
        }
      "))
    ),
    shinyjs::useShinyjs(),
    
    fluidRow(
      column(5, panel(
        tags$h2("Top Action Clients"),
        tags$br(),
        lapply(clientIds, function(id){
          client <- clients[[toString(id)]]
          clientButton(id, client$name, client$income, 
                       paste0("profiles/", client$image))
        })
      ),
      panel(
        tags$h2("Recent Client Activity"),
        tags$br(),
        DTOutput("recentEvents")
      )),
      column(7, 
         panel(
           h2("Revenue Opportunities"),
           ggiraphOutput("revenueOppsPie", width = "600px", height = "400px")
         ),
         panel(
           tags$h2("Market News"),
           tags$br(),
           apply(news, 1, function(article){
             panel(
               h4(article[6]),
               p(paste0(substring(article[7], 0, 400), '...')),
               a(article[5], icon("external-link"), href = article[5], target = "_blank")
             )
           })
         )
      )
    )
  )
}

# Transform events for display on dashboard
displayEvents <- tail(events, 1000)
displayEvents <- displayEvents[sample(nrow(displayEvents), 100),]
displayEvents <- displayEvents[seq(dim(displayEvents)[1],1),]
displayEvents <- merge(event_types, displayEvents, by="EVENT_TYPE_ID")
displayEvents <- displayEvents[,c("CUSTOMER_ID", "EVENT_DATE", "NAME", "CATEGORY")]

revenueOppsData <- data.frame(eventType = c("House Sell", "Equity Buy", "New Sec Account Open", "Other"),
                         value = c(457, 302, 235, 121)) %>%
  mutate(
    percentage = value / sum(value),
    hover_text = paste0(eventType, ": ", round(100 * percentage, 1), "%")
  )
revenueOppsPlot <- ggplot(revenueOppsData, aes(y = value, fill = eventType)) +
  geom_bar_interactive(
    aes(x = 1, tooltip = hover_text),
    stat = "identity",
    show.legend = TRUE
  ) + 
  scale_fill_manual(values = c("House Sell" = "purple", "Equity Buy" = "skyblue", 
                               "New Sec Account Open" = "yellow", "Other" = "blue")) +
  coord_polar(theta = "y") +
  theme_void() +
  theme(legend.title=element_text(size=16), 
        legend.text=element_text(size=12))
revenueOppsPlot <- revenueOppsPlot + guides(fill=guide_legend(title="Event Types"))

homeServer <- function(input, output, session, sessionVars) {
  
  # Table displays for Life Event Prediction Dashboard
  output$recentEvents <- renderDT(displayEvents, rownames = FALSE, style = 'bootstrap')
  
  # Observation events for client buttons
  lapply(paste0('client-btn-', clientIds),
         function(x){
           observeEvent(
             input[[x]],
             {
               id <- as.numeric(sub("client-btn-", "", x))
               sessionVars$selectedClientId <- id
               updateTabsetPanel(session, "lfeNav", selected = "clientPanel")
             }
           )
         })
  
  # Render pie chart of revenue opportunities
  output$revenueOppsPie <- renderggiraph(ggiraph(ggobj = revenueOppsPlot, width_svg=6, height_svg=3))
}
