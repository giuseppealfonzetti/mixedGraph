#'@export
setUp_edge <- function(NODE_I, NODE_J, VAL, EDGEMAT, NODES_TYPE, CUMSUM_NODES_TYPE){
    if(NODE_I < NODE_J){
        #cat("ERROR: interested cells must be in the lower triangular edge matrix!\n")
    }else{
        if(NODE_I <= p & NODE_J <= p){
            EDGEMAT[NODE_I, NODE_J] <- VAL
            #cat(paste0("beta_", NODE_I, NODE_J), "modified.\n")
        }else if(NODE_I > p & NODE_J <= p){
            cat_i <- NODES_TYPE[NODE_I]
            row_start <- CUMSUM_NODES_TYPE[NODE_I-1] + 1
            row_end <- CUMSUM_NODES_TYPE[NODE_I-1] + cat_i
            if(length(VAL) <= cat_i){
                EDGEMAT[row_start:row_end, NODE_J] <- VAL
                #cat(paste0("rho_", NODE_I, NODE_J), "modified.\n")
            } else{
                #cat("ERROR: value object too long!\n")
            }

        }else if(NODE_I > p & NODE_J > p & NODE_I != NODE_J){
            cat_i <- NODES_TYPE[NODE_I]
            row_start <- CUMSUM_NODES_TYPE[NODE_I-1] + 1
            row_end <- CUMSUM_NODES_TYPE[NODE_I-1] + cat_i
            cat_j <- NODES_TYPE[NODE_J]
            col_start <- CUMSUM_NODES_TYPE[NODE_J-1] + 1
            col_end <- CUMSUM_NODES_TYPE[NODE_J-1] + cat_j
            if(length(VAL) <= cat_i*cat_j){
                EDGEMAT[row_start:row_end, col_start:col_end] <- VAL
                #cat(paste0("phi_", NODE_I, NODE_J), "modified.\n")
            } else{
                #cat("ERROR: value object too long!\n")
            }
        }else if(NODE_I > p & NODE_J > p & NODE_I == NODE_J){
            cat_i <- NODES_TYPE[NODE_I]
            row_start <- CUMSUM_NODES_TYPE[NODE_I-1] + 1
            row_end <- CUMSUM_NODES_TYPE[NODE_I-1] + cat_i
            EDGEMAT[row_start:row_end, row_start:row_end] <- diag(VAL, cat_i, cat_i)
            #cat(paste0("phi_", NODE_I, NODE_J), "modified.\n")
        }
    }

    return(EDGEMAT)
}

#'@export
fun_sd <- function(DATA, P, Q){
    vec_sd <- c()
    for (node in 1:P) {
        vec_sd[node] <- sd(as.matrix(DATA[, node]))
    }

    for (node in (P+1):(P+Q)) {
        prob <- as.numeric(table(as.matrix(DATA[, node]))/length(as.matrix(DATA[, node])))
        var <- prob * (1-prob)
        vec_sd[node] <- sqrt(sum(var))
    }
    return(vec_sd)
}

#'@export
recoveryRate <- function(true_edge, est_theta, verb = F){
    estEdge <- theta_to_edgesPar(est_theta, nodes_type, cumsum_nodes_type, p, r, n_nodes)
    tp <- 0; tn <- 0
    TP <- 0; TN <- 0
    for (node_j in 1:(n_nodes-1)) {
        for (node_i in (node_j+1):n_nodes) {
            if(node_i < node_j){
                cat("ERROR: interested cells must be in the lower triangular edge matrix!\n")
            }else{
                if(node_i <= p & node_j <= p){
                    true_val <- true_edge[node_i, node_j]
                    est_val <- estEdge[node_i, node_j]
                    if(true_val == 0){tn = tn+1}
                    if(true_val != 0){tp = tp+1}
                    if(true_val == 0 & est_val == 0){TN = TN+1}
                    if(true_val != 0 & est_val != 0){TP = TP+1}

                    #cat(paste0("beta_", node_i, node_j), "modified.\n")
                }else if(node_i > p & node_j <= p){
                    cat_i <- nodes_type[node_i]
                    row_start <- cumsum_nodes_type[node_i-1] + 1
                    row_end <- cumsum_nodes_type[node_i-1] + cat_i
                    #if(length(val) <= cat_i){
                    true_val <- true_edge[row_start:row_end, node_j]
                    est_val <- estEdge[row_start:row_end, node_j]
                    if(sum(true_val == 0) == 2){tn = tn+1}
                    if(sum(true_val == 0) != 2){tp = tp+1}
                    if((sum(true_val == 0) == 2) & (sum(est_val == 0) == 2)){TN = TN+1}
                    if((sum(true_val == 0) != 2)& (sum(est_val == 0) != 2)){TP = TP+1}

                    #cat(paste0("rho_", node_i, node_j), "modified.\n")
                    # } else{
                    #   cat("ERROR: value object too long!\n")
                    # }
                    #
                }else if(node_i > p & node_j > p & node_i != node_j){
                    cat_i <- nodes_type[node_i]
                    row_start <- cumsum_nodes_type[node_i-1] + 1
                    row_end <- cumsum_nodes_type[node_i-1] + cat_i
                    cat_j <- nodes_type[node_j]
                    col_start <- cumsum_nodes_type[node_j-1] + 1
                    col_end <- cumsum_nodes_type[node_j-1] + cat_j
                    #if(length(val) <= cat_i*cat_j){
                    true_val <- true_edge[row_start:row_end, col_start:col_end]
                    est_val <- estEdge[row_start:row_end, col_start:col_end]
                    if((sum(true_val == 0) == 4)){tn = tn+1}
                    if((sum(true_val == 0) != 4)){tp = tp+1}
                    if((sum(true_val == 0) == 4) & (sum(est_val == 0) == 4)){TN = TN+1}
                    if((sum(true_val == 0) != 4) & (sum(est_val == 0) != 4)){TP = TP+1}
                    #cat(paste0("phi_", node_i, node_j), "modified.\n")
                    # } else{
                    #   cat("ERROR: value object too long!\n")
                    # }
                }else if(node_i > p & node_j > p & node_i == node_j){
                    cat_i <- nodes_type[node_i]
                    row_start <- cumsum_nodes_type[node_i-1] + 1
                    row_end <- cumsum_nodes_type[node_i-1] + cat_i
                    true_val <- true_edge[row_start:row_end, col_start:col_end]
                    est_val <- estEdge[row_start:row_end, col_start:col_end]
                    # if((sum(true_val) == 0) & sum(est_val) == 0){tn = tn+1}
                    # if((true_val != 0) & est_val != 0){tp = tp+1}
                    #cat(paste0("phi_", node_i, node_j), "modified.\n")
                }
            }
        }
    }
    if(verb)cat("TPR:", round(TP/tp, 2), " (",TP,"/",tp, ")", "\nTNR:", round(TN/tn, 2),  " (",TN,"/",tn, ")", "\nTotal:", round((TP+TN)/(tp+tn), 2),  " (",TP+TN,"/",tp+tn, ")\n" )
    return(c(TPR = round(TP/tp, 2), TNR = round(TN/tn, 2), Total = round((TP+TN)/(tp+tn), 2)))
}

#'@export
check_edge <- function(NODE_I, NODE_J, EDGEMAT, NODES_TYPE, CUMSUM_NODES_TYPE){
    out <- FALSE
    if(NODE_I < NODE_J){
        #cat("ERROR: interested cells must be in the lower triangular edge matrix!\n")
    }else{
        if(NODE_I <= p & NODE_J <= p){
            out <- EDGEMAT[NODE_I, NODE_J] != 0
            #cat(paste0("beta_", NODE_I, NODE_J), "modified.\n")
        }else if(NODE_I > p & NODE_J <= p){
            cat_i <- NODES_TYPE[NODE_I]
            row_start <- CUMSUM_NODES_TYPE[NODE_I-1] + 1
            row_end <- CUMSUM_NODES_TYPE[NODE_I-1] + cat_i
                out <- sum(EDGEMAT[row_start:row_end, NODE_J] !=0) > 0
                #cat(paste0("rho_", NODE_I, NODE_J), "modified.\n")


        }else if(NODE_I > p & NODE_J > p & NODE_I != NODE_J){
            cat_i <- NODES_TYPE[NODE_I]
            row_start <- CUMSUM_NODES_TYPE[NODE_I-1] + 1
            row_end <- CUMSUM_NODES_TYPE[NODE_I-1] + cat_i
            cat_j <- NODES_TYPE[NODE_J]
            col_start <- CUMSUM_NODES_TYPE[NODE_J-1] + 1
            col_end <- CUMSUM_NODES_TYPE[NODE_J-1] + cat_j
                out <- sum(EDGEMAT[row_start:row_end, col_start:col_end] != 0) > 0
                #cat(paste0("phi_", NODE_I, NODE_J), "modified.\n")
        }else if(NODE_I > p & NODE_J > p & NODE_I == NODE_J){
            cat_i <- NODES_TYPE[NODE_I]
            row_start <- CUMSUM_NODES_TYPE[NODE_I-1] + 1
            row_end <- CUMSUM_NODES_TYPE[NODE_I-1] + cat_i
            out <- sum(EDGEMAT[row_start:row_end, row_start:row_end] != 0) > 0
            #cat(paste0("phi_", NODE_I, NODE_J), "modified.\n")
        }
    }

    return(out)
}

#'@export
tidy_recovery <- function(TRUE_EDGE, EST_EDGE, P, Q, NODES_TYPE, CUMSUM_NODES_TYPE){
    tmp <- expand_grid(row_node = 1:(P+Q), col_node = 1:(P+Q)) %>%
        filter(col_node < row_node) %>%
        mutate(
            true_edge = map2_lgl(row_node, col_node, ~check_edge(.x,.y, TRUE_EDGE, NODES_TYPE, CUMSUM_NODES_TYPE)),
            est_edge = map2_lgl(row_node, col_node, ~check_edge(.x,.y, EST_EDGE, NODES_TYPE, CUMSUM_NODES_TYPE))
        )



    rates <- tibble(
        tp = tmp %>%
            filter(true_edge==T) %>%
            mutate(tp = (true_edge == est_edge)) %>%
            pluck('tp') %>% mean(),
        tn = tmp %>%
            filter(true_edge==F) %>%
            mutate(tn = (true_edge == est_edge)) %>%
            pluck('tn') %>% mean(),
        tt = tmp %>%
            mutate(tt = (true_edge == est_edge)) %>%
            pluck('tt') %>% mean()
    )

    return(
        list(
            all_edges = tmp,
            rates = rates
        )
    )
}
