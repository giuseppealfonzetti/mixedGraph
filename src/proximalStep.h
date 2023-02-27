#ifndef proximalStep_H
#define proximalStep_H

void beta_softThresholding(
    unsigned node_i,
    unsigned node_j,
    const unsigned int &p,
    const double &lambda,
    const double &gamma,
    Eigen::MatrixXd &edgesPar,
    const Eigen::MatrixXd &D_edgesPar,
    const double &w = 1
){
  // check reference to appropriate parameter
  if(node_i >= p | node_j>= p){
    Rcpp::Rcout << "ERROR: not a continuous edge parameter!\n";
  }else if(node_i == node_j){
    Rcpp::Rcout << "ERROR: not an edge-related parameter!\n";
  }else if(node_j > node_i){
    // Ensure the cell considered is in the lower triangle
    Rcpp::Rcout << "Switching from cell (" << node_i << "," << node_j << ") ";
    const unsigned int node_tmp = node_i;
    node_i = node_j;
    node_j = node_tmp;
    Rcpp::Rcout << "to cell (" << node_i << "," << node_j << ")\n ";
  }
  
  // current value
  const double beta = edgesPar(node_i, node_j);
  const double d = D_edgesPar(node_i, node_j);
  
  // lasso threshold
  const double thr = w *lambda * gamma /d;
  
  // soft-thresholding
  if(beta > thr){
    edgesPar(node_i, node_j) = beta - thr;
  } else if(beta < -thr){
    edgesPar(node_i, node_j) = beta + thr;
  } else {
    edgesPar(node_i, node_j) = 0;
  }
}

void rho_softThresholding(
    const unsigned &node_i,
    const unsigned &node_j,
    const unsigned int &p,
    const double &lambda,
    const double &gamma,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    Eigen::MatrixXd &edgesPar,
    const Eigen::MatrixXd &D_edgesPar,
    const double &w = 1
){
  // check reference to appropriate parameter
  if(node_i < p){
    Rcpp::Rcout << "ERROR: node_i must be categorical!\n";
  }
  if(node_i == node_j){
    Rcpp::Rcout << "ERROR: not an edge-related parameter!\n";
  }
  if(node_j >= p){
    Rcpp::Rcout << "ERROR: node_j must be continuous!\n";
  }
  
  // parameter-block coordinates
  const unsigned int start_row = cumsum_nodes_type[node_i - 1];
  const unsigned int lenght = nodes_type[node_i];
  
  // current value
  const Eigen::VectorXd rho = edgesPar.block(start_row, node_j, lenght, 1);
  const Eigen::VectorXd Drho = D_edgesPar.block(start_row, node_j, lenght, 1);
  const double rhoNorm = Eigen::VectorXd(rho.array()*Drho.array()).norm();
  
  // lasso threshold
  const double thr = w * lambda * gamma ;
  
  // group soft-thresholding
  if(rhoNorm > thr){
    edgesPar.block(start_row, node_j, lenght, 1) = rho.array() - (thr/rhoNorm)*rho.array();
  } else {
    edgesPar.block(start_row, node_j, lenght, 1) = Eigen::VectorXd::Zero(lenght);
  }
  
}

void phi_softThresholding(
    unsigned node_i,
    unsigned node_j,
    const unsigned int &p,
    const double &lambda,
    const double &gamma,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    Eigen::MatrixXd &edgesPar,
    const Eigen::MatrixXd &D_edgesPar,
    const double &w = 1
){
  // check reference to appropriate parameter
  if(node_i < p){
    Rcpp::Rcout << "ERROR: node_i must be categorical!\n";
  }else if(node_i == node_j){
    Rcpp::Rcout << "ERROR: not an edge-related parameter!\n";
  } else if(node_j < p){
    Rcpp::Rcout << "ERROR: node_j must be categorical!\n";
  } else if(node_j > node_i){
    // Ensure the cell considered is in the lower triangle
    Rcpp::Rcout << "Switching from block (" << node_i << "," << node_j << ") ";
    const unsigned int node_tmp = node_i;
    node_i = node_j;
    node_j = node_tmp;
    Rcpp::Rcout << "to block (" << node_i << "," << node_j << ")\n ";
  } else{
    
    // parameter-block coordinates
    const unsigned int start_row = cumsum_nodes_type[node_i - 1];
    const unsigned int start_col = cumsum_nodes_type[node_j - 1];
    const unsigned int nrows = nodes_type[node_i];
    const unsigned int ncols = nodes_type[node_j];
    
    
    // current value
    const Eigen::MatrixXd phi = edgesPar.block(start_row, start_col, nrows, ncols);
    const Eigen::VectorXd Dphi = D_edgesPar.block(start_row, start_col, nrows, ncols);
    const double phiNorm = Eigen::VectorXd(phi.array()*Dphi.array()).norm();
    
    // lasso threshold
    const double thr = w * lambda * gamma;
    
    // group soft-thresholding
    if(phiNorm > thr){
      edgesPar.block(start_row, start_col, nrows, ncols) = phi.array() - (thr/phiNorm)*phi.array();
    } else {
      edgesPar.block(start_row, start_col, nrows, ncols) = Eigen::MatrixXd::Zero(nrows, ncols);
    }
  }
  
  
  
}

// [[Rcpp::export]]
Rcpp::List proximal_stepR(
    const Eigen::VectorXd &theta,
    const Eigen::VectorXd &Dvec,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const unsigned int p,
    const unsigned int r,
    const unsigned int n_nodes,
    const double lambda,
    const double gamma,
    const bool verboseFLAG = false
){
  //Eigen::VectorXd prox_theta = theta;
  Eigen::MatrixXd edgesPar = theta_to_edgesPar(theta, nodes_type, cumsum_nodes_type, p, r, n_nodes);
  Eigen::MatrixXd D_edgesPar = theta_to_edgesPar(Dvec, nodes_type, cumsum_nodes_type, p, r, n_nodes);
  
  // soft-thresholding betas
  for(unsigned int node_i = 1; node_i < p; node_i ++){
    for(unsigned int node_j = 0; node_j < node_i; node_j ++){
      beta_softThresholding(node_i, node_j, p, lambda, gamma, edgesPar, D_edgesPar); 
    }
  }
  
  
  // group soft-thresholding rhos
  for(unsigned int node_i = p; node_i < n_nodes; node_i ++){
    for(unsigned int node_j = 0; node_j < p; node_j++){
      rho_softThresholding(node_i, node_j, p, lambda, gamma, nodes_type, cumsum_nodes_type, edgesPar, D_edgesPar);
    }
  }
  
  // group soft-thresholding phis
  for(unsigned int node_i = p+1; node_i < n_nodes; node_i ++){
    for(unsigned int node_j = p; node_j < node_i; node_j ++){
      phi_softThresholding(node_i, node_j, p, lambda, gamma, nodes_type, cumsum_nodes_type, edgesPar, D_edgesPar);
    }
  }
  
  // convert edge matrix to vector
  std::vector<double> prox_theta;
  for(unsigned int i = 0; i < p; i ++){
    prox_theta.push_back(theta(i));
  }
  
  for(unsigned int j = 0; j < r; j++){
    for(unsigned int i = j; i < r; i++){
      // identify the node
      const unsigned int node_j = count_if(cumsum_nodes_type.begin(), cumsum_nodes_type.end(), [j](int n) { return (n-1) < j; } );
      const unsigned int node_i = count_if(cumsum_nodes_type.begin(), cumsum_nodes_type.end(), [i](int n) { return (n-1) < i; } );
      
      // identify node type
      unsigned int node_j_type = 0; if(nodes_type[node_j]>1) node_j_type = 1;
      unsigned int node_i_type = 0; if(nodes_type[node_i]>1) node_i_type = 1;
      
      // identify par type
      unsigned int par_type = 0; std::string par_type_lab = "beta"; //beta
      if( node_i_type != node_j_type){
        par_type = 1; par_type_lab = "rho";//rho 
      } else if( node_i_type == 1 & node_j_type == 1){
        par_type = 2; par_type_lab = "phi";//phi
      }
      
      // identify category if categorical node
      unsigned int cat_node_j = -1; unsigned int cat_node_i = -1;
      if(node_j_type == 1){
        cat_node_j = j - cumsum_nodes_type[node_j-1];
      }
      if(node_i_type == 1){
        cat_node_i = i - cumsum_nodes_type[node_i-1];
      }
      
      // plug-in vector elements in the matrix
      if(!(par_type == 2 & node_i == node_j & j!=i)){
        prox_theta.push_back(edgesPar(i,j));
      }      
    }
  }
  
  Eigen::Map<Eigen::VectorXd> e_prox_theta(&prox_theta[0], prox_theta.size());
  //edgesPar = edgesPar.selfadjointView<Eigen::Lower>(); // get symmetric edges matrix
  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("edgesPar") = edgesPar,
      Rcpp::Named("st_prox_theta") = prox_theta,
      Rcpp::Named("e_prox_theta") = e_prox_theta
    );
  
  return output;
}

void proximal_step(
    Eigen::VectorXd &theta,
    Eigen::MatrixXd &edgesPar,
    const Eigen::VectorXd &Dvec,
    const std::vector<double> &sds,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    const unsigned int &p,
    const unsigned int &r,
    const unsigned int &n_nodes,
    const double &lambda,
    const double &gamma,
    const bool verboseFLAG = false
){
  
  Eigen::MatrixXd tmp_edgesPar = theta_to_edgesPar(theta, nodes_type, cumsum_nodes_type, p, r, n_nodes);
  Eigen::MatrixXd D_edgesPar = theta_to_edgesPar(Dvec, nodes_type, cumsum_nodes_type, p, r, n_nodes);
  
  // soft-thresholding betas
  for(unsigned int node_i = 1; node_i < p; node_i ++){
    for(unsigned int node_j = 0; node_j < node_i; node_j ++){
      const double w = sds[node_i]*sds[node_j];
      beta_softThresholding(node_i, node_j, p, lambda, gamma, tmp_edgesPar, D_edgesPar, w); 
    }
  }
  
  
  // group soft-thresholding rhos
  for(unsigned int node_i = p; node_i < n_nodes; node_i ++){
    for(unsigned int node_j = 0; node_j < p; node_j++){
      const double w = sds[node_i]*sds[node_j];
      rho_softThresholding(node_i, node_j, p, lambda, gamma, nodes_type, cumsum_nodes_type, tmp_edgesPar, D_edgesPar, w);
    }
  }
  
  // group soft-thresholding phis
  for(unsigned int node_i = p+1; node_i < n_nodes; node_i ++){
    for(unsigned int node_j = p; node_j < node_i; node_j ++){
      const double w = sds[node_i]*sds[node_j];
      phi_softThresholding(node_i, node_j, p, lambda, gamma, nodes_type, cumsum_nodes_type, tmp_edgesPar, D_edgesPar, w);
    }
  }
  
  // extract theta
  std::vector<double> prox_theta;
  
  // nodes parameters
  for(unsigned int i = 0; i < p; i ++){
    prox_theta.push_back(theta(i));
  }
  
  // edges parameters
  for(unsigned int j = 0; j < r; j++){
    for(unsigned int i = j; i < r; i++){
      // identify the node
      const unsigned int node_j = count_if(cumsum_nodes_type.begin(), cumsum_nodes_type.end(), [j](int n) { return (n-1) < j; } );
      const unsigned int node_i = count_if(cumsum_nodes_type.begin(), cumsum_nodes_type.end(), [i](int n) { return (n-1) < i; } );
      
      // identify node type
      unsigned int node_j_type = 0; if(nodes_type[node_j]>1) node_j_type = 1;
      unsigned int node_i_type = 0; if(nodes_type[node_i]>1) node_i_type = 1;
      
      // identify par type
      unsigned int par_type = 0; std::string par_type_lab = "beta"; //beta
      if( node_i_type != node_j_type){
        par_type = 1; par_type_lab = "rho";//rho 
      } else if( node_i_type == 1 & node_j_type == 1){
        par_type = 2; par_type_lab = "phi";//phi
      }
      
      // identify category if categorical node
      unsigned int cat_node_j = -1; unsigned int cat_node_i = -1;
      if(node_j_type == 1){
        cat_node_j = j - cumsum_nodes_type[node_j-1];
      }
      if(node_i_type == 1){
        cat_node_i = i - cumsum_nodes_type[node_i-1];
      }
      
      // plug-in vector elements in the matrix
      if(!(par_type == 2 & node_i == node_j & j!=i)){
        prox_theta.push_back(tmp_edgesPar(i,j));
      }      
    }
  }
  
  theta = Eigen::Map<Eigen::VectorXd> (&prox_theta[0], prox_theta.size());
  edgesPar = tmp_edgesPar;
}

double regularization_term(
    Eigen::VectorXd &theta,
    Eigen::MatrixXd &edgesPar,
    const std::vector<double> &sds,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    const unsigned int &p,
    const unsigned int &r,
    const unsigned int &n_nodes,
    const double &lambda,
    const double &gamma,
    const bool verboseFLAG = false
){
  

  double rt = 0; 
  
  // soft-thresholding betas
  for(unsigned int node_i = 1; node_i < p; node_i ++){
    for(unsigned int node_j = 0; node_j < node_i; node_j ++){
      const double w = sds[node_i]*sds[node_j];
      rt += lambda*w*abs(edgesPar(node_i, node_j)); 
    }
  }
  
  
  // group soft-thresholding rhos
  for(unsigned int node_i = p; node_i < n_nodes; node_i ++){
    for(unsigned int node_j = 0; node_j < p; node_j++){
      const double w = sds[node_i]*sds[node_j];
      // parameter-block coordinates
      const unsigned int start_row = cumsum_nodes_type[node_i - 1];
      const unsigned int lenght = nodes_type[node_i];
      
      // current value
      const Eigen::VectorXd rho = edgesPar.block(start_row, node_j, lenght, 1);
      const double rhoNorm = rho.norm();
      rt += lambda*w*rhoNorm;
      }
  }
  
  // group soft-thresholding phis
  for(unsigned int node_i = p+1; node_i < n_nodes; node_i ++){
    for(unsigned int node_j = p; node_j < node_i; node_j ++){
      const double w = sds[node_i]*sds[node_j];
      const unsigned int start_row = cumsum_nodes_type[node_i - 1];
      const unsigned int start_col = cumsum_nodes_type[node_j - 1];
      const unsigned int nrows = nodes_type[node_i];
      const unsigned int ncols = nodes_type[node_j];
      
      
      // current value
      const Eigen::MatrixXd phi = edgesPar.block(start_row, start_col, nrows, ncols);
      const double phiNorm = phi.norm();
      rt += lambda*w*phiNorm;
      
      }
  }
  
  return rt;
}

#endif
