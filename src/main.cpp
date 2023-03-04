#include <Rcpp.h>
#define EIGEN_PERMANENTLY_DISABLE_STUPID_WARNINGS
#define EIGEN_DONT_PARALLELIZE
#include <RcppEigen.h>
// #include <RcppParallel.h>
#include <RcppClock.h>
#include <random>
#include <math.h>
#include <unistd.h>

#include "utils.h"
#include "continuousNodes.h"
#include "categoricalNodes.h"
#include "proximalStep.h"
#include "sampler.h"


// [[Rcpp::depends(RcppEigen, RcppParallel, RcppClock)]]

//' @export
// [[Rcpp::export]]
Rcpp::List graph_cl(
    const Eigen::MatrixXd &DATA,
    const Eigen::VectorXd &THETA,
    const std::vector<unsigned int> NODES_TYPE,
    const bool VERBOSEFLAG = false,
    const bool GRADFLAG = false,
    const bool GRAD2FLAG = false

){

  // Identify dimensions
  const unsigned int n = DATA.rows();                                // number of units
  const unsigned int n_nodes = NODES_TYPE.size();                    // number of nodes
  unsigned int p = 0;                                                // number of continuous nodes
  for( unsigned int node = 0; node < NODES_TYPE.size(); node++){
    if(NODES_TYPE[node] == 1) p++; else break;
  }
  const unsigned int q = n_nodes - p;                                // number of categorical nodes
  const unsigned int r = std::accumulate(NODES_TYPE.begin(), NODES_TYPE.end(), decltype(NODES_TYPE)::value_type(0)); // dimension of edges parameter matrix

  // cumultative sum of NODES_TYPE. Used to identify parameters in edgesPar
  std::vector<unsigned int> cumsum_nodes_type = NODES_TYPE;
  for( unsigned int node = 1; node < n_nodes; node++) cumsum_nodes_type[node] = cumsum_nodes_type[node-1] + cumsum_nodes_type[node];

  // Initialize objective function and gradient
  double cl = 0;
  Eigen::VectorXd gradient = Eigen::VectorXd::Zero(THETA.size());
  Eigen::VectorXd dHess = Eigen::VectorXd::Zero(THETA.size());

  // Build nodes-parameter vector and edges-parameter matrix
  Eigen::VectorXd nodesPar = theta_to_nodesPar(THETA, NODES_TYPE, p);
  Eigen::MatrixXd edgesPar = theta_to_edgesPar(THETA, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes);

  // double loop over units and nodes
  for(unsigned int i = 0; i < n; i++){
    const Eigen::VectorXd datai = DATA.row(i);

    for(unsigned int node = 0; node < n_nodes; node++){

      // continuous node or catagorical
      if(node < p){
        const double Ei = E_continuous_node(datai, edgesPar, nodesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
        const double xi = datai(node);
        const double node_l = logp_continuous_node(Ei, edgesPar(node, node), xi);
        cl += node_l;
        if(GRADFLAG)gradient += gradient_continuous_node(Ei, edgesPar(node, node), xi, p, node, edgesPar.cols(), datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
        if(GRAD2FLAG)dHess += dHess_continuous_node(Ei, edgesPar(node, node), xi, p, node, edgesPar.cols(), datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);


      }else{
        const double node_l = logp_categorical_node(datai, edgesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
        cl += node_l;
        std::vector<double> probs(NODES_TYPE[node]);
        if(GRADFLAG)gradient += gradientV2_categorical_node(datai, edgesPar, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
        if(GRAD2FLAG)dHess += dHess_categorical_node(datai, edgesPar, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);

      }
    }
  }

  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("cl") = cl,
      Rcpp::Named("gradient") = gradient,
      Rcpp::Named("dHess") = dHess,
      Rcpp::Named("nodesPar") = nodesPar,
      Rcpp::Named("edgesPar") = edgesPar,
      Rcpp::Named("p") = p,
      Rcpp::Named("q") = q
    );

  return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List mixedGraph_old(
    const Eigen::MatrixXd &DATA,
    const std::vector<double> &SDS,
    const Eigen::VectorXd &THETA,
    const std::vector<unsigned int> NODES_TYPE,
    const unsigned int MAXITER,
    double STEPSIZE,
    const double REG_PAR,
    const double NU = 1,
    const double TOL = 1e-4,
    const unsigned int TOL_MINCOUNT = 4,
    const bool DHESSFLAG = false,
    const bool VERBOSEFLAG = false,
    const bool REGFLAG = true,
    const unsigned int BURN = 25,
    const unsigned int SAMPLING_SCHEME = 1, // 0 CL, 1 Be-CL, 2 mini-Batch SGD, 3 mini-BATCH hyper-SGD
    const unsigned int BATCH = 1, // sgd BATCH size
    const unsigned int SEED = 123
){

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("Main");

  // Identify dimensions
  const unsigned int n = DATA.rows();                                // number of units
  const unsigned int n_nodes = NODES_TYPE.size();                    // number of nodes
  unsigned int p = 0;                                                // number of continuous nodes
  for( unsigned int node = 0; node < NODES_TYPE.size(); node++){
    if(NODES_TYPE[node] == 1) p++; else break;
  }
  const unsigned int q = n_nodes - p;                                // number of categorical nodes
  const unsigned int r = std::accumulate(NODES_TYPE.begin(), NODES_TYPE.end(), decltype(NODES_TYPE)::value_type(0)); // dimension of edges parameter matrix

  // cumultative sum of NODES_TYPE. Used to identify parameters in edgesPar
  std::vector<unsigned int> cumsum_nodes_type = NODES_TYPE;
  for( unsigned int node = 1; node < n_nodes; node++) cumsum_nodes_type[node] = cumsum_nodes_type[node-1] + cumsum_nodes_type[node];



  // scaling constant according to sampling_scheme
  // Compute scaling constant
  double scale;
  switch(SAMPLING_SCHEME){
  case 0: // numeric
    scale = 1/static_cast<double>(n) ;
    break;
  case 1: // sgd
    scale = 1/static_cast<double>(NU);
    break;
  case 2: // bernoulli
    scale = 1/static_cast<double>(NU);
    break;
  case 3: // hypergeometric
    scale = 1/static_cast<double>(NU);
    break;
  }
  Rcpp::Rcout << "  Final scale = " << scale<< "\n";


  // Initialize parameter vector and gradient
  Eigen::VectorXd iter_theta = THETA;

  // Initialize storage for iterations quantities
  Eigen::MatrixXd path_theta    = Eigen::MatrixXd::Zero(MAXITER +1, THETA.size()); path_theta.row(0) = THETA;
  Eigen::MatrixXd path_av_theta = Eigen::MatrixXd::Zero(MAXITER +1, THETA.size()); path_theta.row(0) = THETA;
  Eigen::MatrixXd path_grad     = Eigen::MatrixXd::Zero(MAXITER,    THETA.size());
  Eigen::VectorXd path_nll(MAXITER);
  Eigen::VectorXd path_regTerm(MAXITER);
  Eigen::VectorXd path_thetaDiff(MAXITER);
  Eigen::VectorXd path_thetaNorm(MAXITER+1); path_thetaNorm(0) = THETA.norm();
  Eigen::VectorXd path_theta_check(MAXITER);
  Eigen::MatrixXd path_theta_diff  = Eigen::MatrixXd::Zero(MAXITER-1, THETA.size());

  // Initialize vector of indexes for entries in pairs_table
  // Set-up the randomizer
  std::vector<int> sublik_pool(n*n_nodes) ;
  std::iota (std::begin(sublik_pool), std::end(sublik_pool), 0);
  std::vector<int> vector_weights(n*n_nodes);

  // Convergence-related quantities
  bool convergence = false;
  unsigned int last_iter = MAXITER;
  unsigned int tol_counter = 0;

  /* OPTIMIZATION LOOP */
  for(unsigned int t = 1; t <= MAXITER; t++){
    // check user interruption
    Rcpp::checkUserInterrupt();
    Rcpp::Rcout << "\rIteration:" << t << " ";
    clock.tick("Iteration");
    // Initialize iteration quantities
    double iter_cl = 0;
    Eigen::VectorXd iter_gradient = Eigen::VectorXd::Zero(THETA.size());
    Eigen::VectorXd iter_dHess = Eigen::VectorXd::Ones(THETA.size());
    if(DHESSFLAG)iter_dHess.setZero();

    // Build nodes-parameter vector
    Eigen::VectorXd iter_nodesPar = theta_to_nodesPar(iter_theta, NODES_TYPE, p);
    Eigen::MatrixXd iter_edgesPar = theta_to_edgesPar(iter_theta, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes);

    // Select units according to SAMPLING_SCHEME
    // Set-up the randomizer

    double prob;
    Eigen::VectorXd w_units = Eigen::VectorXd::Ones(n);
    if(SAMPLING_SCHEME == 2){
      std::mt19937 randomizer(SEED+t);
      std::vector<int> units(n) ;
      std::iota (std::begin(units), std::end(units), 0);
      std::shuffle(units.begin(), units.end(), randomizer);
      w_units = Eigen::VectorXd::Zero(n);
      for(unsigned int i = 0; i < BATCH; i ++){
        w_units(units[i]) = 1;
      }
    };

    // double loop over units and nodes to compute gradient and obj
    clock.tick("Iteration gradient");
    for(unsigned int i = 0; i < n; i++){

        //Rcpp::Rcout<< "t = "<< t << ", i = " << i << ", node: ";
        const Eigen::VectorXd datai = DATA.row(i);

        for(unsigned int node = 0; node < n_nodes; node++){

          unsigned int weight;
          switch(SAMPLING_SCHEME){
          case 0: // numeric
            weight = 1;
            break;
          case 1: // sgd

            if(w_units[i]==1) weight = 1;
            break;
          case 2: // bernoulli
            prob = static_cast<double>(NU)/static_cast<double>(n);
            if(R::runif(0,1) < prob ) weight = 1;
          }



          if(weight != 0){
            // continuous node or catagorical
            if(node < p){
              const double Ei = E_continuous_node(datai, iter_edgesPar, iter_nodesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
              const double xi = datai(node);
              const double node_l = logp_continuous_node(Ei, iter_edgesPar(node, node), xi);
              iter_cl -= node_l;
              iter_gradient -= gradient_continuous_node(Ei, iter_edgesPar(node, node), xi, p, node, r, datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
              // if(DHESSFLAG)iter_dHess -= dHess_continuous_node(Ei, iter_edgesPar(node, node), xi, p, node, r, datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);


            }else{
              const double node_l = logp_categorical_node(datai, iter_edgesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
              iter_cl -= node_l;
              std::vector<double> probs(NODES_TYPE[node]);
              iter_gradient -= gradientV2_categorical_node(datai, iter_edgesPar, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
              // if(DHESSFLAG)iter_dHess -= dHess_categorical_node(datai, iter_edgesPar, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
            }
          }
        }

      }

    iter_cl *= scale;
    iter_gradient *= scale;
    clock.tock("Iteration gradient");


    // regularization term for convergence checks
    const double iter_rt = regularization_term(iter_theta, iter_edgesPar, SDS, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes, REG_PAR, STEPSIZE, VERBOSEFLAG);
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", theta: "; if(VERBOSEFLAG)print_vec(iter_theta);

    // 1st order theta update
    clock.tick("Iteration update");
    double stepsize = STEPSIZE*pow(t+1, -.5-1e-2);
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", stepsize: "<< stepsize<< "\n";
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", gr: "; if(VERBOSEFLAG)print_vec(iter_gradient);

    //if(!DHESSFLAG)iter_gradient/=n;
    if(DHESSFLAG)iter_dHess =  (iter_dHess.array()/(NU) + 1e-10);
    Eigen::VectorXd iter_D = iter_dHess.array().inverse();
    iter_theta -= stepsize * iter_D.asDiagonal() * iter_gradient;
    clock.tock("Iteration update");

    // proximal step
    clock.tick("Iteration proximal step");
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t+1 << ", theta pre thresholding: "; if(VERBOSEFLAG)print_vec(iter_theta);
    Eigen::VectorXd iter_D2 = Eigen::VectorXd::Ones(THETA.size());
    if(REGFLAG)proximal_step(iter_theta, iter_edgesPar, iter_D, SDS, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes, REG_PAR, stepsize, VERBOSEFLAG);
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t+1 << ", theta post thresholding: "; if(VERBOSEFLAG)print_vec(iter_theta);
    clock.tock("Iteration proximal step");



    // store iter quantities
    path_theta.row(t) = iter_theta;
    path_thetaNorm(t) = iter_theta.norm();
    path_grad.row(t-1) = iter_gradient;
    // path_dHess.row(t-1) = iter_dHess;
    path_nll(t-1) = iter_cl;
    path_regTerm(t-1) = iter_rt;

    // check convergence
      const Eigen::VectorXd theta_diff = path_theta.row(t) - path_theta.row(t-1);
      const double theta_diff_norm = theta_diff.norm();
      const double theta_diff_check = theta_diff_norm/path_thetaNorm(t-1);

      path_thetaDiff(t-1) = theta_diff_norm;
      path_theta_check(t-1) = theta_diff_check;
      path_theta_diff.row(t-1) = theta_diff;

      Rcpp::Rcout << " check:" << theta_diff_check << " ";
      if( theta_diff_check <= TOL){
        tol_counter++;
        if(tol_counter == TOL_MINCOUNT){
          convergence = true;
          last_iter = t;
          break;
        }
      }else{
        tol_counter = 0;
      }


    clock.tock("Iteration");

  }

  path_theta.conservativeResize(last_iter+2, THETA.size());
  path_grad.conservativeResize(last_iter, THETA.size());
  // path_dHess.conservativeResize(last_iter, THETA.size());
  path_nll.conservativeResize(last_iter);
  path_regTerm.conservativeResize(last_iter);
  path_thetaDiff.conservativeResize(last_iter);
  path_theta_check.conservativeResize(last_iter);
  path_thetaNorm.conservativeResize(last_iter+2);

  clock.tock("Main");
  clock.stop("clock");

  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("burn") = BURN,
      Rcpp::Named("p") = p,
      Rcpp::Named("q") = q,
      Rcpp::Named("path_theta") = path_theta,
      Rcpp::Named("path_av_theta") = path_av_theta,
      Rcpp::Named("path_grad") = path_grad,
      // Rcpp::Named("path_dHess") = path_dHess,
      Rcpp::Named("path_nll") = path_nll,
      Rcpp::Named("path_regTerm") = path_regTerm,
      Rcpp::Named("path_thetaDiff") = path_thetaDiff,
      Rcpp::Named("path_thetaNorm") = path_thetaNorm,
      Rcpp::Named("path_theta_check") = path_theta_check,
      Rcpp::Named("path_theta_diff") = path_theta_diff,
      Rcpp::Named("convergence") = convergence,
      Rcpp::Named("last_iter") = last_iter
    );

  return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List mixedGraph(
    const Eigen::MatrixXd &DATA,
    const std::vector<double> &SDS,
    const Eigen::VectorXd &THETA,
    const std::vector<unsigned int> NODES_TYPE,
    const unsigned int MAXITER,
    double STEPSIZE,
    const double REG_PAR,
    const double NU = 1,
    const double TOL = 1e-4,
    const unsigned int TOL_MINCOUNT = 4,
    const bool VERBOSEFLAG = false,
    const bool REGFLAG = true,
    const unsigned int BURN = 25,
    const unsigned int SAMPLING_SCHEME = 1, // 0 CL, 1 Be-CL, 2 mini-Batch SGD, 3 mini-BATCH hyper-SGD
    const unsigned int SEED = 123,
    const double EPS = .55
){

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("Main");

  // Identify dimensions
  const unsigned int n = DATA.rows();                                // number of units
  const unsigned int n_nodes = NODES_TYPE.size();                    // number of nodes
  unsigned int p = 0;                                                // number of continuous nodes
  for( unsigned int node = 0; node < NODES_TYPE.size(); node++){
    if(NODES_TYPE[node] == 1) p++; else break;
  }
  const unsigned int q = n_nodes - p;                                // number of categorical nodes
  const unsigned int r = std::accumulate(NODES_TYPE.begin(), NODES_TYPE.end(), decltype(NODES_TYPE)::value_type(0)); // dimension of edges parameter matrix

  // cumultative sum of nodes_type. Used to identify parameters in edgesPar
  std::vector<unsigned int> cumsum_nodes_type = NODES_TYPE;
  for( unsigned int node = 1; node < n_nodes; node++) cumsum_nodes_type[node] = cumsum_nodes_type[node-1] + cumsum_nodes_type[node];

  double prob = static_cast<double>(NU)/static_cast<double>(n);

  // scaling constant according to sampling_scheme
  // Compute scaling constant
  double scale;
  switch(SAMPLING_SCHEME){
  case 0: // numeric
    scale = 1/static_cast<double>(n) ;
    break;
  case 1: // sgd
    scale = 1/static_cast<double>(NU);
    break;
  case 2: // bernoulli
    scale = 1/static_cast<double>(NU);
    break;
  case 3: // hypergeometric
    scale = 1/static_cast<double>(NU);
    break;
  }  Rcpp::Rcout << " Final scale = " << scale<< "\n";
  /* OPTIMIZATION LOOP */

  // Initialize paraeter vector and gradient
  Eigen::VectorXd iter_theta = THETA;

  // Initialize storage for iterations quantities
  std::vector<Eigen::VectorXd> path_theta; path_theta.push_back(THETA);
  std::vector<Eigen::VectorXd> path_grad;
  std::vector<double> path_nll;
  std::vector<double> path_check;
  std::vector<double> path_norm; path_norm.push_back(THETA.norm());

  // Convergence-related quantities
  bool convergence = false;
  unsigned int last_iter = MAXITER;
  unsigned int tol_counter = 0;


  for(unsigned int t = 0; t < MAXITER; t++){
    Rcpp::checkUserInterrupt();
    Rcpp::Rcout << "\rIteration:" << t << " ";

    // Rcpp::Rcout << ", prob:" << prob << " ";
    clock.tick("Iteration");

    // Initialize iteration quantities
    double iter_cl = 0;
    Eigen::VectorXd iter_gradient = Eigen::VectorXd::Zero(THETA.size());

    // Build nodes-parameter vector
    Eigen::VectorXd iter_nodesPar = theta_to_nodesPar(iter_theta, NODES_TYPE, p);
    Eigen::MatrixXd iter_edgesPar = theta_to_edgesPar(iter_theta, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes);

    // sgd sampling scheme
    Eigen::VectorXd w_units = Eigen::VectorXd::Ones(n);
    if(SAMPLING_SCHEME == 1){
      std::mt19937 randomizer(SEED+t);
      std::vector<int> units(n) ;
      std::iota (std::begin(units), std::end(units), 0);
      std::shuffle(units.begin(), units.end(), randomizer);
      w_units = Eigen::VectorXd::Zero(n);
      for(unsigned int i = 0; i < NU; i ++){
        w_units(units[i]) = 1;
      }
    }

    // double loop over units and nodes to compute gradient and obj
    // clock.tick("Iteration gradient");
    for(unsigned int i = 0; i < n; i++){

      if(w_units(i)==1){
        //Rcpp::Rcout<< "t = "<< t << ", i = " << i << ", node: ";
        const Eigen::VectorXd datai = DATA.row(i);


        Eigen::VectorXd w_nodes = Eigen::VectorXd::Ones(n_nodes);
        if(SAMPLING_SCHEME == 2){
          for(unsigned int node = 0; node < n_nodes; node ++){
            w_nodes(node) = R::rbinom(1, prob);
          }
        }


        for(unsigned int node = 0; node < n_nodes; node++){

          const unsigned int sel = w_nodes(node);
          if(sel == 1){
            //Rcpp::Rcout<< node << ", ";
            // continuous node or categorical
            if(node < p){
              const double Ei = E_continuous_node(datai, iter_edgesPar, iter_nodesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
              const double xi = datai(node);
              const double node_l = logp_continuous_node(Ei, iter_edgesPar(node, node), xi);
              iter_cl -= node_l;
              iter_gradient -= gradient_continuous_node(Ei, iter_edgesPar(node, node), xi, p, node, r, datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
            }else{
              const double node_l = logp_categorical_node(datai, iter_edgesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
              iter_cl -= node_l;
              std::vector<double> probs(NODES_TYPE[node]);
              iter_gradient -= gradientV2_categorical_node(datai, iter_edgesPar, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
            }
          }

        }

        //Rcpp::Rcout<<"\n";
      }

    }
    iter_gradient*= scale;
    // clock.tock("Iteration gradient");


    // regularization term for convergence checks
    // const double iter_rt = regularization_term(iter_theta, iter_edgesPar, SDS, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes, REG_PAR, STEPSIZE, VERBOSEFLAG);
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", theta: "; if(VERBOSEFLAG)print_vec(iter_theta);

    // 1st order theta update
    // clock.tick("Iteration update");
    double stepsize = STEPSIZE*pow(t+1, -EPS);
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", stepsize: "<< stepsize<< "\n";
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", gr: "; if(VERBOSEFLAG)print_vec(iter_gradient);

    Eigen::VectorXd iter_D = Eigen::VectorXd::Ones(THETA.size());
    iter_theta -= stepsize * iter_gradient;
    // clock.tock("Iteration update");

    // proximal step
    // clock.tick("Iteration proximal step");
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t+1 << ", theta pre thresholding: "; if(VERBOSEFLAG)print_vec(iter_theta);
    Eigen::VectorXd iter_D2 = Eigen::VectorXd::Ones(THETA.size());
    if(REGFLAG)proximal_step(iter_theta, iter_edgesPar, iter_D, SDS, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes, REG_PAR, stepsize, VERBOSEFLAG);
    if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t+1 << ", theta post thresholding: "; if(VERBOSEFLAG)print_vec(iter_theta);
    // clock.tock("Iteration proximal step");



    // store iter quantities
    path_theta.push_back(iter_theta);
    path_grad.push_back(iter_gradient);

    path_nll.push_back(iter_cl*scale);
    path_norm.push_back(iter_theta.norm());

    // check convergence
    const Eigen::VectorXd theta_diff = path_theta[t+1] - path_theta[t];
    const double theta_diff_norm = theta_diff.norm();
    const double theta_diff_check = theta_diff_norm/path_norm[t];
    path_check.push_back(theta_diff_check);

    Rcpp::Rcout << " check:" << theta_diff_check << " ";

    if(theta_diff_check <= TOL){
      tol_counter++;
      if(tol_counter == TOL_MINCOUNT){
        convergence = true;
        last_iter = t;
        break;
      }
    }else{
      tol_counter = 0;
    }


    clock.tock("Iteration");

  }

  clock.tock("Main");
  clock.stop("clock");

  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("burn") = BURN,
      Rcpp::Named("p") = p,
      Rcpp::Named("q") = q,
      Rcpp::Named("path_theta") = path_theta,
      Rcpp::Named("path_grad") = path_grad,
      Rcpp::Named("path_nll") = path_nll,
      Rcpp::Named("path_norm") = path_norm,
      Rcpp::Named("path_check") = path_check,
      Rcpp::Named("convergence") = convergence,
      Rcpp::Named("last_iter") = last_iter
    );

  return output;
}

//' @export
// [[Rcpp::export]]
Rcpp::List mixedGraph_SVRG(
    const Eigen::MatrixXd &DATA,
    const std::vector<double> &SDS,
    const Eigen::VectorXd &THETA,
    const std::vector<unsigned int> NODES_TYPE,
    const unsigned int MAXITER,
    double STEPSIZE,
    const double REG_PAR,
    const unsigned int M,
    const double NU = 1,
    const double TOL = 1e-4,
    const unsigned int TOL_MINCOUNT = 4,
    const bool VERBOSEFLAG = false,
    const bool REGFLAG = true,
    const unsigned int BURN = 25,
    const unsigned int SAMPLING_SCHEME = 1, // 0 CL, 1 Be-CL, 2 mini-Batch SGD, 3 mini-BATCH hyper-SGD
    const unsigned int SEED = 123,
    const double EPS = .55
){

  // Set up clock monitor to export to R session trough RcppClock
  Rcpp::Clock clock;
  clock.tick("Main");

  // Identify dimensions
  const unsigned int n = DATA.rows();                                // number of units
  const unsigned int n_nodes = NODES_TYPE.size();                    // number of nodes
  unsigned int p = 0;                                                // number of continuous nodes
  for( unsigned int node = 0; node < NODES_TYPE.size(); node++){
    if(NODES_TYPE[node] == 1) p++; else break;
  }
  const unsigned int q = n_nodes - p;                                // number of categorical nodes
  const unsigned int r = std::accumulate(NODES_TYPE.begin(), NODES_TYPE.end(), decltype(NODES_TYPE)::value_type(0)); // dimension of edges parameter matrix

  // cumultative sum of nodes_type. Used to identify parameters in edgesPar
  std::vector<unsigned int> cumsum_nodes_type = NODES_TYPE;
  for( unsigned int node = 1; node < n_nodes; node++) cumsum_nodes_type[node] = cumsum_nodes_type[node-1] + cumsum_nodes_type[node];

  double prob = static_cast<double>(NU)/static_cast<double>(n);

  // scaling constant according to sampling_scheme
  // Compute scaling constant
  double scale;
  switch(SAMPLING_SCHEME){
  case 0: // numeric
    scale = 1/static_cast<double>(n) ;
    break;
  case 1: // sgd
    scale = 1/static_cast<double>(NU);
    break;
  case 2: // bernoulli
    scale = 1/static_cast<double>(NU);
    break;
  case 3: // hypergeometric
    scale = 1/static_cast<double>(NU);
    break;
  }  Rcpp::Rcout << " Final scale = " << scale<< "\n";
  /* OPTIMIZATION LOOP */

  // Initialize paraeter vector and gradient
  Eigen::VectorXd iter_theta = THETA;

  // Initialize storage for iterations quantities
  std::vector<Eigen::VectorXd> path_theta; path_theta.push_back(THETA);
  std::vector<Eigen::VectorXd> path_grad;
  std::vector<Eigen::VectorXd> path_grad_ref;
  std::vector<Eigen::VectorXd> path_grad_iter_ref;
  std::vector<Eigen::VectorXd> path_grad_SVRG;


  std::vector<double> path_nll;
  std::vector<double> path_check;
  std::vector<double> path_norm; path_norm.push_back(THETA.norm());

  // Convergence-related quantities
  bool convergence = false;
  unsigned int last_iter = MAXITER;
  unsigned int tol_counter = 0;


  Eigen::VectorXd theta_ref   = Eigen::VectorXd::Zero(THETA.size());
  Eigen::VectorXd theta_init  = Eigen::VectorXd::Zero(THETA.size());
  Eigen::VectorXd theta_s     = THETA;

  // OUTER LOOP
  for(unsigned int s = 0; s < MAXITER; s++){

    Rcpp::checkUserInterrupt();
    Rcpp::Rcout << "\nEpoch:" << s << "\n";

    theta_ref = theta_s;
    Eigen::VectorXd grad_ref = Eigen::VectorXd::Zero(THETA.size());
    iter_theta = theta_ref;

    // Build nodes-parameter vector
    Eigen::VectorXd nodesPar_ref = theta_to_nodesPar(theta_ref, NODES_TYPE, p);
    Eigen::MatrixXd edgesPar_ref = theta_to_edgesPar(theta_ref, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes);

    // Full gradient once each M steps -> grad_ref
    Rcpp::Rcout << "1. Computing full gradient...\n";
    for(unsigned int i = 0; i < n; i++){

      const Eigen::VectorXd datai = DATA.row(i);
      for(unsigned int node = 0; node < n_nodes; node++){

        //Rcpp::Rcout<< node << ", ";
        // continuous node or catagorical
        if(node < p){
          const double Ei = E_continuous_node(datai, edgesPar_ref, nodesPar_ref, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
          const double xi = datai(node);
          grad_ref -= gradient_continuous_node(Ei, edgesPar_ref(node, node), xi, p, node, r, datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
        }else{
          std::vector<double> probs(NODES_TYPE[node]);
          grad_ref -= gradientV2_categorical_node(datai, edgesPar_ref, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
        }
      }
    }
    grad_ref /= n;

    path_grad_ref.push_back(grad_ref);
    // iter_theta -= 1 * grad_ref;
    // path_theta.push_back(iter_theta);


    // INNER LOOP:
    // Rcpp::Rcout <<
    for(unsigned int t = 0; t < M; t++){

      Rcpp::checkUserInterrupt();
      Rcpp::Rcout << "\r2. Inner loop. Step "<< t << ".";
      // sleep(1);

        // Initialize iteration quantities
        double iter_cl = 0;
        Eigen::VectorXd iter_gradient = Eigen::VectorXd::Zero(THETA.size());
        Eigen::VectorXd iter_grad_ref = Eigen::VectorXd::Zero(THETA.size());
        Eigen::VectorXd iter_gradient_SVRG = Eigen::VectorXd::Zero(THETA.size());
        Eigen::VectorXd iter_D = Eigen::VectorXd::Ones(THETA.size());


        // Build nodes-parameter vector
        Eigen::VectorXd iter_nodesPar = theta_to_nodesPar(iter_theta, NODES_TYPE, p);
        Eigen::MatrixXd iter_edgesPar = theta_to_edgesPar(iter_theta, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes);

        // sgd sampling scheme
        Eigen::VectorXd w_units = Eigen::VectorXd::Ones(n);
        if(SAMPLING_SCHEME == 1){
          std::mt19937 randomizer(SEED+t+s);
          std::vector<int> units(n) ;
          std::iota (std::begin(units), std::end(units), 0);
          std::shuffle(units.begin(), units.end(), randomizer);
          w_units = Eigen::VectorXd::Zero(n);
          for(unsigned int i = 0; i < NU; i ++){
            w_units(units[i]) = 1;
          }
        }

        // double loop over units and nodes to compute gradient and obj
        // clock.tick("Iteration gradient");
        for(unsigned int i = 0; i < n; i++){

          if(w_units(i)==1){
            //Rcpp::Rcout<< "t = "<< t << ", i = " << i << ", node: ";
            const Eigen::VectorXd datai = DATA.row(i);


            Eigen::VectorXd w_nodes = Eigen::VectorXd::Ones(n_nodes);
            if(SAMPLING_SCHEME == 2){
              for(unsigned int node = 0; node < n_nodes; node ++){
                w_nodes(node) = R::rbinom(1, prob);
              }
            }


            for(unsigned int node = 0; node < n_nodes; node++){

              const unsigned int sel = w_nodes(node);
              if(sel == 1){
                //Rcpp::Rcout<< node << ", ";
                // switch continuous node or categorical
                if(node < p){

                  // approximation at theta_t
                  const double xi = datai(node);
                  double Ei = E_continuous_node(datai, iter_edgesPar, iter_nodesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
                  const double node_l = logp_continuous_node(Ei, iter_edgesPar(node, node), xi);
                  iter_cl -= node_l;
                  iter_gradient -= gradient_continuous_node(Ei, iter_edgesPar(node, node), xi, p, node, r, datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);

                  // approximation at theta_ref
                  Ei = E_continuous_node(datai, edgesPar_ref, nodesPar_ref, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
                  iter_grad_ref -= gradient_continuous_node(Ei, edgesPar_ref(node, node), xi, p, node, r, datai, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);

                }else{

                  // approximation at theta_t
                  const double node_l = logp_categorical_node(datai, iter_edgesPar, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);
                  iter_cl -= node_l;
                  std::vector<double> probs(NODES_TYPE[node]);
                  iter_gradient -= gradientV2_categorical_node(datai, iter_edgesPar, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);

                  // approximation at theta_ref
                  iter_grad_ref -= gradientV2_categorical_node(datai, edgesPar_ref, probs, node, p, NODES_TYPE, cumsum_nodes_type, VERBOSEFLAG);

                }
              }

            }

            //Rcpp::Rcout<<"\n";
          }

        }
        iter_grad_ref*=scale;
        iter_gradient*=scale;
        // clock.tock("Iteration gradient");


        // regularization term for convergence checks
        const double iter_rt = regularization_term(iter_theta, iter_edgesPar, SDS, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes, REG_PAR, STEPSIZE, VERBOSEFLAG);
        if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", theta: "; if(VERBOSEFLAG)print_vec(iter_theta);

        // 1st order theta update
        // clock.tick("Iteration update");
        double stepsize = STEPSIZE*pow(t+1, -EPS);
        if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", stepsize: "<< stepsize<< "\n";
        if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t << ", gr: "; if(VERBOSEFLAG)print_vec(iter_gradient);

        iter_gradient_SVRG = grad_ref + iter_gradient - iter_grad_ref;

        // iter_theta -= stepsize * iter_gradient;
         iter_theta -= stepsize * iter_gradient_SVRG;
        // clock.tock("Iteration update");

        // proximal step
        // clock.tick("Iteration proximal step");
        if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t+1 << ", theta pre thresholding: "; if(VERBOSEFLAG)print_vec(iter_theta);
        if(REGFLAG)proximal_step(iter_theta, iter_edgesPar, iter_D, SDS, NODES_TYPE, cumsum_nodes_type, p, r, n_nodes, REG_PAR, stepsize, VERBOSEFLAG);
        if(VERBOSEFLAG)Rcpp::Rcout<< "t = "<< t+1 << ", theta post thresholding: "; if(VERBOSEFLAG)print_vec(iter_theta);
        // clock.tock("Iteration proximal step");

        // store iter quantities
        path_theta.push_back(iter_theta);
        path_grad_SVRG.push_back(iter_gradient_SVRG);
        path_grad.push_back(iter_gradient);
        path_grad_iter_ref.push_back(iter_grad_ref);

        path_nll.push_back(iter_cl*scale);
        path_norm.push_back(iter_theta.norm());

        // check convergence
        const Eigen::VectorXd theta_diff = path_theta[t+1] - path_theta[t];
        const double theta_diff_norm = theta_diff.norm();
        const double theta_diff_check = theta_diff_norm/path_norm[t];
        path_check.push_back(theta_diff_check);


    }

    theta_s = iter_theta;

  }



  clock.tock("Main");
  clock.stop("clock");

  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("burn") = BURN,
      Rcpp::Named("p") = p,
      Rcpp::Named("q") = q,
      Rcpp::Named("path_theta") = path_theta,
      // Rcpp::Named("path_av_theta") = path_av_theta,
      Rcpp::Named("path_grad") = path_grad,
      Rcpp::Named("path_grad_SVRG") = path_grad_SVRG,
      Rcpp::Named("path_grad_ref") = path_grad_ref,
      Rcpp::Named("path_grad_iter_ref") = path_grad_iter_ref,

      // Rcpp::Named("path_dHess") = path_dHess,
      Rcpp::Named("path_nll") = path_nll,
      // Rcpp::Named("path_regTerm") = path_regTerm,
      // Rcpp::Named("path_thetaDiff") = path_thetaDiff,
      Rcpp::Named("path_norm") = path_norm,
      Rcpp::Named("path_check") = path_check,
      // Rcpp::Named("path_theta_diff") = path_theta_diff,
      Rcpp::Named("convergence") = convergence,
      Rcpp::Named("last_iter") = last_iter
    );

  return output;
}










































