#ifndef categoricalNodes_H
#define categoricalNodes_H

// [[Rcpp::export]]
const double eta_categorical_node(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const unsigned int r,
    const unsigned int p,
    const unsigned int cat,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  
  double eta = 0;
  if(r < p){
    Rcpp::Rcout << "ERROR: Node " << r << " is continuous!\n";
  } else if(cat >= nodes_type[r]){
    Rcpp::Rcout << "ERROR: Node " << r << " has not "<< cat + 1 <<" categories!\n";
  }else{
    const unsigned int row = cumsum_nodes_type[r-1] + cat;
    
    if(verboseFLAG)Rcpp::Rcout << "Obs category on chosen node:" << cat << "\n";
    for( unsigned int node = 0; node < data.size(); node ++){
      
      // rho: categorical - continuous
      if(node < p){
        if(verboseFLAG)Rcpp::Rcout << " |_ Node " << node<< ", val: " << data(node) << ", edgePar: rho "<< edgesPar(row, node) <<" \n";
        
        eta += edgesPar(row, node) * data(node);
      }
      
      // phirr: 
      if(node == r){
        if(verboseFLAG)Rcpp::Rcout << " |_ Node " << node<< ", category: " << data(node) << ", edgePar: phirr "<< edgesPar(row, row) <<"\n";
        eta += edgesPar(row, row);
      }
      
      // phirj: categorical - categorical
      if(node >= p & node != r){
        const unsigned int obs_cat = data(node);
        if(r > node){
          if(verboseFLAG)Rcpp::Rcout << " |_ Node " << node<< ", category: " << data(node) << ", col " << cumsum_nodes_type[node -1] + obs_cat;
          if(verboseFLAG)Rcpp::Rcout << ", edgePar: phirj "<< edgesPar(row, cumsum_nodes_type[node-1] + obs_cat) <<"\n";
          eta += edgesPar(row, cumsum_nodes_type[node-1] + obs_cat);    
          }else{
            if(verboseFLAG)Rcpp::Rcout << " |_ Node " << node<< ", category: " << data(node) << ", col " << cumsum_nodes_type[node -1] + obs_cat;
            if(verboseFLAG)Rcpp::Rcout << ", edgePar: phirj "<< edgesPar(cumsum_nodes_type[node-1] + obs_cat, row) <<"\n";
            eta += edgesPar(cumsum_nodes_type[node-1] + obs_cat, row);    
            }
      
      

    }
    
    }
  }
  
  
  
  return eta;
}

// [[Rcpp::export]]
const double logp_categorical_node(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const unsigned int r,
    const unsigned int p,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  
  double ll = 0;
  if(r < p){
    Rcpp::Rcout << "Node " << r << " is continuous!\n";
  }else{
    const unsigned int yr = data(r);
    const unsigned int row = cumsum_nodes_type[r-1] + yr;
    
    const double eta = eta_categorical_node(data, edgesPar, r, p, yr, nodes_type, cumsum_nodes_type, verboseFLAG);
    
    double norm_const = 0;
    if(verboseFLAG)Rcpp::Rcout << "Computing normalizing constant:\n";
    for(unsigned int cat = 0; cat < nodes_type[r]; cat ++){
      const double eta_cat = eta_categorical_node(data, edgesPar, r, p, cat, nodes_type, cumsum_nodes_type, verboseFLAG);
      norm_const += exp(eta_cat);
      
    }
    
    ll = log(exp(eta)/norm_const); 
  }
  
  return ll;
}

// [[Rcpp::export]]
Eigen::VectorXd gradient_categorical_node(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const unsigned int r,
    const unsigned int p,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const double prob,
    const bool verboseFLAG = false
){
  std::vector<double> std_grad;
  const double dim_edgesPar = edgesPar.cols();
  
  
  for( unsigned int j = 0; j < p; j++){
    double dalpha = 0;
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
        double prob_catl = 0;
        if(node_j == r | node_i == r){ // relevant parameter
          
          // rho
          if(par_type == 1){
            
            if(node_j == r & data(node_j) == cat_node_j){
              der = (1 - prob)*data(node_i);
              if(verboseFLAG)Rcpp::Rcout << " --> rho";
              
            }else if (node_i == r & data(node_i) == cat_node_i){
              der = (1 - prob)*data(node_j);
              if(verboseFLAG)Rcpp::Rcout << " --> rho";
              
            }else if(node_i == r & data(node_i) != cat_node_i){
              Eigen::VectorXd data_cat = data; data_cat(node_i) = cat_node_i;
              double l_cat = logp_categorical_node(data_cat, edgesPar, r,  p, nodes_type, cumsum_nodes_type, verboseFLAG);
              der = -exp(l_cat)*data(node_j);
            }else if(node_j == r & data(node_j) != cat_node_j){
              Eigen::VectorXd data_cat = data; data_cat(node_j) = cat_node_j;
              double l_cat = logp_categorical_node(data_cat, edgesPar, r,  p, nodes_type, cumsum_nodes_type, verboseFLAG);
              der = -exp(l_cat)*data(node_i);
            }
          }else if(node_j == node_i & par_type == 2 & cat_node_i == data(node_i)){
            // phi_rr
            if(verboseFLAG)Rcpp::Rcout << " --> phi_rr";
            der = 1 - prob;
          }else if(node_j == node_i & par_type == 2 & cat_node_i != data(node_i)){
            // phi_rr
            if(verboseFLAG)Rcpp::Rcout << " --> phi_rr";
            Eigen::VectorXd data_cat = data; data_cat(node_i) = cat_node_i;
            double l_cat = logp_categorical_node(data_cat, edgesPar, r,  p, nodes_type, cumsum_nodes_type, verboseFLAG);
            der = -exp(l_cat);

          }else if(par_type == 2 & node_j != node_i){
            // phi_rs  
            if(cat_node_i == data(node_i) & cat_node_j == data(node_j)){
              if(verboseFLAG)Rcpp::Rcout << " --> phi_rt";
              
              der = 1 - prob;
            }else if(node_i == r & cat_node_i != data(node_i) & cat_node_j == data(node_j)){
              Eigen::VectorXd data_cat = data; data_cat(node_i) = cat_node_i;
              double l_cat = logp_categorical_node(data_cat, edgesPar, r,  p, nodes_type, cumsum_nodes_type, verboseFLAG);
              der = -exp(l_cat);
            }else if(node_j == r & cat_node_j != data(node_j) & cat_node_i == data(node_i)){
              Eigen::VectorXd data_cat = data; data_cat(node_j) = cat_node_j;
              double l_cat = logp_categorical_node(data_cat, edgesPar, r,  p, nodes_type, cumsum_nodes_type, verboseFLAG);
              der = -exp(l_cat);
            }
            
          }
        }
        
        if(verboseFLAG)Rcpp::Rcout << "\n";
        if(!(par_type==2 & node_i == node_j & cat_node_i != cat_node_j)) std_grad.push_back(der);
        
        
      }
    }
  }  
  
  return Eigen::Map<Eigen::VectorXd> (&std_grad[0], std_grad.size());
}

// [[Rcpp::export]]
Eigen::VectorXd gradientV2_categorical_node(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    std::vector<double> &probs,
    const unsigned int r,
    const unsigned int p,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  std::vector<double> std_grad;
  const double dim_edgesPar = edgesPar.cols();
  const unsigned int cat_r = nodes_type[r];
  
  //Rcpp::Rcout << "probs : ";
  for( unsigned int s = 0; s < cat_r; s++){
    Eigen::VectorXd data_cat = data; data_cat(r) = s;
    double l_cat = logp_categorical_node(data_cat, edgesPar, r,  p, nodes_type, cumsum_nodes_type, verboseFLAG);
    probs[s] = exp(l_cat);
    //Rcpp::Rcout << probs[s] << ", ";
  }
  //Rcpp::Rcout << "\n";
  
  
  for( unsigned int j = 0; j < p; j++){
    double dalpha = 0;
    std_grad.push_back(dalpha);
  }
  
  for( unsigned int j = 0; j < dim_edgesPar; j++){
    for( unsigned int i = j; i < dim_edgesPar; i++){
      
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
        double prob_catl = 0;
        if(node_j == r | node_i == r){ // relevant parameter
          
          // rho
          if(par_type == 1){
            
            if(node_j == r){
              unsigned int indr = 0;
              if(data(node_j) == cat_node_j)indr = 1;
              der = (indr-probs[cat_node_j])*data(node_i);
              if(verboseFLAG)Rcpp::Rcout << " --> rho";
              
            }else if (node_i == r){
              unsigned int indr = 0;
              if(data(node_i) == cat_node_i)indr = 1;
              der = (indr-probs[cat_node_i])*data(node_j);
              if(verboseFLAG)Rcpp::Rcout << " --> rho";
            }
          }else if(node_j == node_i & par_type == 2){
            // phi_rr
            unsigned int indr = 0;
            if(data(node_i) == cat_node_i)indr = 1;
            if(verboseFLAG)Rcpp::Rcout << " --> phi_rr";
            der = indr - probs[cat_node_i];
          }else if(par_type == 2 & node_j != node_i){
            // phi_rs  
            if(node_i == r){
              unsigned int indr = 0;
              unsigned int indj = 0;
              if(data(node_i) == cat_node_i)indr = 1;
              if(data(node_j) == cat_node_j)indj = 1;
              
              der = indr*indj - indj*probs[cat_node_i];
            }else if(node_j == r){
              unsigned int indr = 0;
              unsigned int indj = 0;
              if(data(node_i) == cat_node_i)indj = 1;
              if(data(node_j) == cat_node_j)indr = 1;
              
              der = indr*indj - indj*probs[cat_node_j];
            }
            
          }
        }
        
        if(verboseFLAG)Rcpp::Rcout << "\n";
        if(!(par_type==2 & node_i == node_j & cat_node_i != cat_node_j)) std_grad.push_back(der);
        
        
      
    }
  }  
  return Eigen::Map<Eigen::VectorXd> (&std_grad[0], std_grad.size());
}
// [[Rcpp::export]]
Eigen::VectorXd dHess_categorical_node(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const std::vector<double> &probs,
    const unsigned int r,
    const unsigned int p,
    const std::vector<unsigned int> nodes_type,
    const std::vector<unsigned int> cumsum_nodes_type,
    const bool verboseFLAG = false
){
  std::vector<double> std_grad;
  const double dim_edgesPar = edgesPar.cols();
  const unsigned int cat_r = nodes_type[r];

  
  
  for( unsigned int j = 0; j < p; j++){
    double dalpha = 0;
    std_grad.push_back(dalpha);
  }
  
  for( unsigned int j = 0; j < dim_edgesPar; j++){
    for( unsigned int i = j; i < dim_edgesPar; i++){
      
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
      double prob_catl = 0;
      if(node_j == r | node_i == r){ // relevant parameter
        
        // rho
        if(par_type == 1){
          
          if(node_j == r){
            der = pow(data(node_i), 2) * (pow(probs[cat_node_j], 2)-probs[cat_node_j]);
            if(verboseFLAG)Rcpp::Rcout << " --> rho";
            
          }else if (node_i == r){
            der = pow(data(node_j), 2) * (pow(probs[cat_node_i], 2)-probs[cat_node_i]);
            
            if(verboseFLAG)Rcpp::Rcout << " --> rho";
          }
        }else if(node_j == node_i & par_type == 2){
          // phi_rr
          der =  (pow(probs[cat_node_i], 2)-probs[cat_node_i]);
        }else if(par_type == 2 & node_j != node_i){
          // phi_rs  
          if(node_i == r){
            unsigned int indj = 0;
            if(data(node_j) == cat_node_j)indj = 1;
            der = indj*(pow(probs[cat_node_i], 2)-probs[cat_node_i]);
          }else if(node_j == r){
            unsigned int indj = 0;
            if(data(node_i) == cat_node_i)indj = 1;
            der = indj*(pow(probs[cat_node_j], 2)-probs[cat_node_j]);
          }
          
        }
      }
      
      if(verboseFLAG)Rcpp::Rcout << "\n";
      if(!(par_type==2 & node_i == node_j & cat_node_i != cat_node_j)) std_grad.push_back(der);
      
      
      
    }
  }  
  return Eigen::Map<Eigen::VectorXd> (&std_grad[0], std_grad.size());
}
#endif

