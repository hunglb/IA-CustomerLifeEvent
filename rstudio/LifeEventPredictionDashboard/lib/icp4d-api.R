# Sample Materials, provided under license.
# Licensed Materials - Property of IBM
# © Copyright IBM Corp. 2019. All Rights Reserved.
# US Government Users Restricted Rights - Use, duplication or disclosure restricted by GSA ADP Schedule Contract with IBM Corp.

# Inputs
# hostname <- 'myicpdcluster.example.com'
# username <- 'myusername@example.com'
# password <- 'mypassword'

collectDeployments <- function(hostname, username, password, target_asset) {
  base_url <- sprintf('https://%s:31843/icp4d-api/v1', hostname)
  
  creds <- list(username=username, password=password)
  authResponse <- content(POST(url=paste(base_url,'/authorize',sep=''),
                               config(ssl_verifyhost = FALSE, ssl_verifypeer = FALSE),
                               body=creds,
                               encode="json",
                               timeout(4)
  ))
  if("token" %in% names(authResponse)) {
    token <- authResponse$token
  } else if("message" %in% names(authResponse)) {
    stop(paste("ERROR:", authResponse$message))
  } else {
    stop(paste("ERROR:", authResponse))
  }
  
  releases <- content(GET(url=paste(base_url,'/releases',sep=''),
                          config(ssl_verifyhost = FALSE, ssl_verifypeer = FALSE),
                          add_headers(Authorization = paste('Bearer', token)),
                          encode="json"
  ))$releases
  
  validDeployments <- list()
  errorMsg <- list("ERROR: No valid deployments found.", " ")
  for(release in releases) {
    deployments <- content(GET(url = sprintf('%s/releases/%s/deployments', base_url, release$release_route),
                               config(ssl_verifyhost = FALSE, ssl_verifypeer = FALSE),
                               add_headers(Authorization = paste('Bearer', token)),
                               encode = "json"
    ))$deployments
    
    for(deployment in deployments) {
      
      # select only deployments with the correct deployment_asset_name
      if(deployment$deployment_asset_name == target_asset) {
        
        # build deployment details URL
        deploymentUrl = sprintf('%s/releases/%s/deployments/%s?deployment_manifest_type=%s&deployment_asset_name=%s',
                                base_url, release$release_route, deployment$deployment_name, deployment$deployment_manifest_type,
                                strsplit(deployment$deployment_asset_name, '.', fixed = TRUE)[[1]][1])
        
        # get deployment details
        deploymentDetails <- content(GET(url = deploymentUrl,
                                          config(ssl_verifyhost = FALSE, ssl_verifypeer = FALSE),
                                          add_headers(Authorization = paste('Bearer', token)),
                                          encode = "json"))$deployment
        
        # add release name to deployment details
        deploymentDetails$release_name <- release$release_name
        
        # deployment URL corrections
        deploymentUrl <- gsub("dsx-core-svc", hostname, deploymentDetails$deployment_url$url)
        deploymentUrl <- gsub("3000", "31843", deploymentUrl)
        deploymentUrl <- gsub(paste0("/dsvc/v1/",release$release_name), paste0("/dsvc/v1/",deploymentDetails$release_route), deploymentUrl)
        
        # deployment URL function specification
        deploymentUrl <- gsub("\\{OPTIONS\\}", 'score', deploymentUrl)
        deploymentUrl <- gsub("\\{FUNCTION_NAME\\}", 'score', deploymentUrl)
        
        deploymentDetails$scoring_url <- deploymentUrl
        
        if("deployment_token" %in% names(deploymentDetails)) {
          validDeployments[[deployment$deployment_name]] <- deploymentDetails
        } else {
          errorMsg <- c(errorMsg, paste0("No deployment token found for deployment '", deploymentDetails$deployment_name, "' in release '", deploymentDetails$release_name, "'. You are likely a Viewer of this project release."), " ")
        }
      }
    }
  }
  if(length(validDeployments) == 0) {
    errorMsg <- c(errorMsg, paste0("Make sure you have Developer or Admin access to a Project Release with a deployment of '",target_asset,"'"))
    stop(paste(errorMsg, collapse='\n'))
  }
  return(validDeployments)
}

scoreModelDeployment <- function(endpoint, token, data) {
  
  resp <- POST(url = endpoint,
               config(ssl_verifyhost = FALSE, ssl_verifypeer = FALSE),
               add_headers(Authorization = token),
               body = list(args = data),
               encode = "json")
  tryCatch({
    json <- fromJSON(content(resp))
    if(length(json$result) > 0) {
      return(json)
    } else if(length(json$stderr) > 0) {
      return(list(error=json$stderr))
    } else {
      return(list(error=json))
    }
  }, error = function(e) {
    return(list(error=content(resp)))
  })
}
