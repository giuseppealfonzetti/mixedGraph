# Generated by using Rcpp::compileAttributes() -> do not edit by hand
# Generator token: 10BE3573-1514-4C36-9D1C-5A225CD40393

eta_categorical_node <- function(data, edgesPar, r, p, cat, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_eta_categorical_node`, data, edgesPar, r, p, cat, nodes_type, cumsum_nodes_type, verboseFLAG)
}

logp_categorical_node <- function(data, edgesPar, r, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_logp_categorical_node`, data, edgesPar, r, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

gradient_categorical_node <- function(data, edgesPar, r, p, nodes_type, cumsum_nodes_type, prob, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_gradient_categorical_node`, data, edgesPar, r, p, nodes_type, cumsum_nodes_type, prob, verboseFLAG)
}

gradientV2_categorical_node <- function(data, edgesPar, probs, r, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_gradientV2_categorical_node`, data, edgesPar, probs, r, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

dHess_categorical_node <- function(data, edgesPar, probs, r, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_dHess_categorical_node`, data, edgesPar, probs, r, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

E_continuous_node <- function(data, edgesPar, nodesPar, s, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_E_continuous_node`, data, edgesPar, nodesPar, s, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

logp_continuous_node <- function(E, betass, xi) {
    .Call(`_mixedGraph_logp_continuous_node`, E, betass, xi)
}

gradient_continuous_node <- function(E, betass, xi, p, s, dim_edgesPar, data, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_gradient_continuous_node`, E, betass, xi, p, s, dim_edgesPar, data, nodes_type, cumsum_nodes_type, verboseFLAG)
}

dHess_continuous_node <- function(E, betass, xi, p, s, dim_edgesPar, data, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_dHess_continuous_node`, E, betass, xi, p, s, dim_edgesPar, data, nodes_type, cumsum_nodes_type, verboseFLAG)
}

#' @export
graph_cl <- function(DATA, THETA, NODES_TYPE, VERBOSEFLAG = FALSE, GRADFLAG = FALSE, GRAD2FLAG = FALSE) {
    .Call(`_mixedGraph_graph_cl`, DATA, THETA, NODES_TYPE, VERBOSEFLAG, GRADFLAG, GRAD2FLAG)
}

#' @export
mixedGraph_old <- function(DATA, SDS, THETA, NODES_TYPE, MAXITER, STEPSIZE, REG_PAR, NU = 1, TOL = 1e-4, TOL_MINCOUNT = 4L, DHESSFLAG = FALSE, VERBOSEFLAG = FALSE, REGFLAG = TRUE, BURN = 25L, SAMPLING_SCHEME = 1L, BATCH = 1L, SEED = 123L) {
    .Call(`_mixedGraph_mixedGraph_old`, DATA, SDS, THETA, NODES_TYPE, MAXITER, STEPSIZE, REG_PAR, NU, TOL, TOL_MINCOUNT, DHESSFLAG, VERBOSEFLAG, REGFLAG, BURN, SAMPLING_SCHEME, BATCH, SEED)
}

#' @export
mixedGraph <- function(DATA, SDS, THETA, NODES_TYPE, MAXITER, STEPSIZE, REG_PAR, NU = 1, TOL = 1e-4, TOL_MINCOUNT = 4L, VERBOSEFLAG = FALSE, REGFLAG = TRUE, BURN = 25L, SAMPLING_SCHEME = 1L, SEED = 123L, EPS = .55) {
    .Call(`_mixedGraph_mixedGraph`, DATA, SDS, THETA, NODES_TYPE, MAXITER, STEPSIZE, REG_PAR, NU, TOL, TOL_MINCOUNT, VERBOSEFLAG, REGFLAG, BURN, SAMPLING_SCHEME, SEED, EPS)
}

#' @export
mixedGraph_SVRG <- function(DATA, SDS, THETA, NODES_TYPE, MAXITER, STEPSIZE, REG_PAR, M, NU = 1, TOL = 1e-4, TOL_MINCOUNT = 4L, VERBOSEFLAG = FALSE, REGFLAG = TRUE, BURN = 25L, SAMPLING_SCHEME = 1L, SEED = 123L, EPS = .55) {
    .Call(`_mixedGraph_mixedGraph_SVRG`, DATA, SDS, THETA, NODES_TYPE, MAXITER, STEPSIZE, REG_PAR, M, NU, TOL, TOL_MINCOUNT, VERBOSEFLAG, REGFLAG, BURN, SAMPLING_SCHEME, SEED, EPS)
}

proximal_stepR <- function(theta, Dvec, nodes_type, cumsum_nodes_type, p, r, n_nodes, lambda, gamma, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_proximal_stepR`, theta, Dvec, nodes_type, cumsum_nodes_type, p, r, n_nodes, lambda, gamma, verboseFLAG)
}

intMultinom <- function(E_probs, verboseFLAG = FALSE, tol = 1e-5) {
    .Call(`_mixedGraph_intMultinom`, E_probs, verboseFLAG, tol)
}

drawCatNode <- function(data, edgesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_drawCatNode`, data, edgesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

drawContNode <- function(data, edgesPar, nodesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_drawContNode`, data, edgesPar, nodesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

graphGibbsSampler <- function(edgesPar, nodesPar, nodes_type, warmup, maxiter, m, skip, verboseFLAG = FALSE, store_warmupFLAG = FALSE) {
    .Call(`_mixedGraph_graphGibbsSampler`, edgesPar, nodesPar, nodes_type, warmup, maxiter, m, skip, verboseFLAG, store_warmupFLAG)
}

cont_gamma <- function(edgesPar, nodesPar, cat_pattern, p, nodes_type, cumsum_nodes_type, verboseFLAG = FALSE) {
    .Call(`_mixedGraph_cont_gamma`, edgesPar, nodesPar, cat_pattern, p, nodes_type, cumsum_nodes_type, verboseFLAG)
}

#' @export
graphBlocksGibbsSampler <- function(edgesPar, nodesPar, nodes_type, warmup, maxiter, m, skip, verboseFLAG = FALSE, store_warmupFLAG = FALSE) {
    .Call(`_mixedGraph_graphBlocksGibbsSampler`, edgesPar, nodesPar, nodes_type, warmup, maxiter, m, skip, verboseFLAG, store_warmupFLAG)
}

#' @export
theta_to_edgesPar <- function(THETA, NODES_TYPE, CUMSUM_NODES_TYPE, P, R, N_NODES) {
    .Call(`_mixedGraph_theta_to_edgesPar`, THETA, NODES_TYPE, CUMSUM_NODES_TYPE, P, R, N_NODES)
}

#' @export
theta_to_edgesParSparse <- function(theta, nodes_type, cumsum_nodes_type, p, r, n_nodes) {
    .Call(`_mixedGraph_theta_to_edgesParSparse`, theta, nodes_type, cumsum_nodes_type, p, r, n_nodes)
}

#' @export
theta_to_nodesPar <- function(theta, nodes_type, p) {
    .Call(`_mixedGraph_theta_to_nodesPar`, theta, nodes_type, p)
}

#' @export
compute_scale <- function(sampling_scheme, n, n_nodes, prob, m, batch) {
    .Call(`_mixedGraph_compute_scale`, sampling_scheme, n, n_nodes, prob, m, batch)
}

#'@export
rmultinom_wrapper <- function(prob, classes, batch, K) {
    .Call(`_mixedGraph_rmultinom_wrapper`, prob, classes, batch, K)
}

