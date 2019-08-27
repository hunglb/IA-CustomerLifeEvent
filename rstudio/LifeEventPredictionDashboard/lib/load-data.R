# Sample Materials, provided under license.
# Licensed Materials - Property of IBM
# © Copyright IBM Corp. 2019. All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

# Load Data from CSV files

library(scales)

readDataset <- function(fileName) { readr::read_csv(file.path("..", "..", "datasets", fileName)) }

customer <- readDataset("customer.csv")
events <- readDataset("event.csv")
event_types <- readDataset("event_type.csv")
news <- readr::read_delim(file.path("..", "..", "datasets", "news.csv"), '|')[2:6,]

clients <- list(
  list(name="Paige Carson", image="1F.jpg"),
  list(name="Alex Anderson", image="2M.jpg"),
  list(name="Ian Gray", image="3M.jpg"),
  list(name="Jane Wilson", image="8F.jpg"),
  list(name="Robert Taylor", image="4M.jpg")
)
clientIds <- c(1039:1041, 1023, 1011)
names(clients) <- clientIds

for(id in clientIds) {
  clients[[toString(id)]]$income <- dollar(customer[customer$CUSTOMER_ID == id,][[1,'ANNUAL_INCOME']])
}
