#ifndef continuousNodes_H
#define continuousNodes_H

// [[Rcpp::export]]
const double E_continuous_node(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const Eigen::VectorXd &nodesPar,
    const unsigned int s,
    const unsigned int p,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  
  double output = 0;
  if(s>=p){
    Rcpp::Rcout << "ERROR: Node " << s << " is not continuous!\n";
  }else{
    const double xs = data(s);
    const double alphas = nodesPar(s);
    const double betass = edgesPar(s, s);
    
    double eta = alphas;
    for(unsigned int node = 0; node < data.size(); node++){
      
      // beta: continuous - continuous
      if(node != s & node < p){
        if( s > node){
          if(verboseFLAG)Rcpp::Rcout << "ROW-WALKING: Node " << node<< ", val: " << data(node) << ", edgePar: "<< edgesPar(s, node) <<"\n";
          eta -= edgesPar(s, node) * data(node);
        }else{
          if(verboseFLAG)Rcpp::Rcout << "COL-WALKING: Node " << node<< ", val: " << data(node) << ", edgePar: "<< edgesPar(node, s) <<"\n";
          eta -= edgesPar(node, s) * data(node);
        }

      }
      
      // rho: continuous - categorical
      if(node >= p){
        const unsigned int obs_cat = data(node);
        if(verboseFLAG)Rcpp::Rcout << "Node " << node<< ", category: " << data(node) << ", col " << cumsum_nodes_type[node -1] + obs_cat;
        if(verboseFLAG)Rcpp::Rcout << ", edgePar: "<< edgesPar(cumsum_nodes_type[node-1] + obs_cat, s) <<"\n";
        
        eta += edgesPar(cumsum_nodes_type[node-1] + obs_cat, s);
        
      }
    }
    
    output = eta/betass;
  }
  
  return output;
}

// [[Rcpp::export]]
const double logp_continuous_node(
    const double &E,
    const double &betass,
    const double &xi
){
  
  const double std = sqrt(1/betass);
  const double ll = R::dnorm( xi, E, std, true );
  
  return ll;
}

// [[Rcpp::export]]
Eigen::VectorXd gradient_continuous_node(
    const double &E,
    const double &betass,
    const double &xi,
    const unsigned int p,
    const unsigned int s,
    const unsigned int dim_edgesPar,
    const Eigen::VectorXd &data,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  
  const double std = sqrt(1/betass);
  const double ll = R::dnorm( xi, E, std, true );
  std::vector<double> std_grad;
  
  for( unsigned int j = 0; j < p; j++){
    double dalpha = 0;
    if(j == s) dalpha = xi - E;
    
    std_grad.push_back(dalpha);
  }
  
  for( unsigned int j = 0; j < dim_edgesPar; j++){
    for( unsigned int i = 0; i < dim_edgesPar; i++){
      
      if (j <= i){
        
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
        
        
        
        
        //build parameter label if needed
        std::string par_lab = par_type_lab + "_" + std::to_string(node_i) + std::to_string(node_j);
        if( par_type == 1){
          par_lab += "(" + std::to_string(cat_node_i) + ")";
        } else if(par_type == 2){
          if(node_i == node_j & j!=i){
            par_lab = "Non relevant";
          }else{
            par_lab += "(" + std::to_string(cat_node_i) + ","+ std::to_string(cat_node_j) +")";
          }
        }
        
        //////////////
        // GRADIENT 
        /////////////
        
        if(verboseFLAG)Rcpp::Rcout <<"("<<i<<","<<j<<"), " <<"Node i:" << node_i << ", type "<< node_i_type << ". Node j:" << node_j<< ", type "<< node_j_type;
        if(verboseFLAG)Rcpp::Rcout << " ----> par: " << par_lab ; 
        
        double der = 0;
        if(node_j == s | node_i == s){
          if(node_j != node_i & par_type == 0){
            // beta_st
            if(node_j== s){
              der = (E - data(node_j))*data(node_i);
            }else if (node_i== s){
              der = (E - data(node_i))*data(node_j);
            }
            if(verboseFLAG)Rcpp::Rcout << " --> beta_st";
          }else if(j == i & par_type == 0){
            // beta_ss
            if(verboseFLAG)Rcpp::Rcout << " --> beta_ss";
            der = .5*((1/betass)+ pow(E, 2)- pow(xi, 2));
          }else if(par_type == 1 & cat_node_i == data(node_i)){
            // rho  
            if(verboseFLAG)Rcpp::Rcout << " --> rho: " << " cat_i:"<< cat_node_i << ", obs:" << data(node_i);
            der = xi - E;
            
          }
        }
        
        if(verboseFLAG)Rcpp::Rcout << "\n";
        if(!(par_type==2 & node_i == node_j & cat_node_i != cat_node_j)) std_grad.push_back(der);        
        
      }
    }
  }  
  
  //Eigen::Map<Eigen::VectorXd> egrad(&std_grad[0], std_grad.size());
  return Eigen::Map<Eigen::VectorXd> (&std_grad[0], std_grad.size());
}

// [[Rcpp::export]]
Eigen::VectorXd dHess_continuous_node(
    const double &E,
    const double &betass,
    const double &xi,
    const unsigned int p,
    const unsigned int s,
    const unsigned int dim_edgesPar,
    const Eigen::VectorXd &data,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  
  const double std = sqrt(1/betass);
  const double ll = R::dnorm( xi, E, std, true );
  std::vector<double> std_grad;
  
  for( unsigned int j = 0; j < p; j++){
    double dalpha = 0;
    if(j == s) dalpha = -1/betass;
    
    std_grad.push_back(dalpha);
  }
  
  for( unsigned int j = 0; j < dim_edgesPar; j++){
    for( unsigned int i = 0; i < dim_edgesPar; i++){
      
      if (j <= i){
        
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
        
        
        
        
        //build parameter label if needed
        std::string par_lab = par_type_lab + "_" + std::to_string(node_i) + std::to_string(node_j);
        if( par_type == 1){
          par_lab += "(" + std::to_string(cat_node_i) + ")";
        } else if(par_type == 2){
          if(node_i == node_j & j!=i){
            par_lab = "Non relevant";
          }else{
            par_lab += "(" + std::to_string(cat_node_i) + ","+ std::to_string(cat_node_j) +")";
          }
        }
        
        //////////////
        // GRADIENT 
        /////////////
        
        if(verboseFLAG)Rcpp::Rcout <<"("<<i<<","<<j<<"), " <<"Node i:" << node_i << ", type "<< node_i_type << ". Node j:" << node_j<< ", type "<< node_j_type;
        if(verboseFLAG)Rcpp::Rcout << " ----> par: " << par_lab ; 
        
        double der = 0;
        if(node_j == s | node_i == s){
          if(node_j != node_i & par_type == 0){
            // beta_st
            if(node_j== s){
              der = -pow(data(node_i), 2)/betass;
            }else if (node_i== s){
              der = -pow(data(node_j), 2)/betass;;
            }
            if(verboseFLAG)Rcpp::Rcout << " --> beta_st";
          }else if(j == i & par_type == 0){
            // beta_ss
            if(verboseFLAG)Rcpp::Rcout << " --> beta_ss";
            der = -.5*pow(betass, -2) - pow(E, 2)/betass;
          }else if(par_type == 1 & cat_node_i == data(node_i)){
            // rho  
            if(verboseFLAG)Rcpp::Rcout << " --> rho: " << " cat_i:"<< cat_node_i << ", obs:" << data(node_i);
            der = -1/betass;
            
          }
        }
        
        if(verboseFLAG)Rcpp::Rcout << "\n";
        if(!(par_type==2 & node_i == node_j & cat_node_i != cat_node_j)) std_grad.push_back(der);        
        
      }
    }
  }  
  
  //Eigen::Map<Eigen::VectorXd> egrad(&std_grad[0], std_grad.size());
  return Eigen::Map<Eigen::VectorXd> (&std_grad[0], std_grad.size());
}
#endif
