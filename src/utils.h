#ifndef utils_H
#define utils_H

/* Function to print vector to rsession in a tidy way*/
void print_vec(Eigen::VectorXd vector){
  Rcpp::Rcout<<"(";
  for(int i = 0; i < vector.size(); i++){
    Rcpp::Rcout << vector(i);
    if(i != vector.size()-1) Rcpp::Rcout<<", ";
  }
  Rcpp::Rcout<<")\n";
}

//' @export
// [[Rcpp::export]]
const Eigen::MatrixXd theta_to_edgesPar(
    const Eigen::VectorXd &THETA,
    const std::vector<unsigned int> &NODES_TYPE,
    const std::vector<unsigned int> &CUMSUM_NODES_TYPE,
    const unsigned int P,
    const unsigned int R,
    const unsigned int N_NODES
){

  Eigen::MatrixXd edgesMat = Eigen::MatrixXd::Zero(R,R);
  unsigned int iterator = P;

  for( unsigned int j = 0; j < R; j++){
    for( unsigned int i = 0; i < R; i++){

      if (j <= i){

        // identify the node
        const unsigned int node_j = count_if(CUMSUM_NODES_TYPE.begin(), CUMSUM_NODES_TYPE.end(), [j](int n) { return (n-1) < j; } );
        const unsigned int node_i = count_if(CUMSUM_NODES_TYPE.begin(), CUMSUM_NODES_TYPE.end(), [i](int n) { return (n-1) < i; } );

        // identify node type
        unsigned int node_j_type = 0; if(NODES_TYPE[node_j]>1) node_j_type = 1;
        unsigned int node_i_type = 0; if(NODES_TYPE[node_i]>1) node_i_type = 1;

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
          cat_node_j = j - CUMSUM_NODES_TYPE[node_j-1];
        }
        if(node_i_type == 1){
          cat_node_i = i - CUMSUM_NODES_TYPE[node_i-1];
        }

        // plug-in vector elements in the matrix
        if(!(par_type == 2 & node_i == node_j & j!=i)){
          edgesMat(i, j) = THETA(iterator);

          iterator ++;
        }


      }
    }
  }


  return(edgesMat);

}

//' @export
// [[Rcpp::export]]
const Eigen::SparseMatrix<double> theta_to_edgesParSparse(
    const Eigen::VectorXd &theta,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    const unsigned int p,
    const unsigned int r,
    const unsigned int n_nodes
){

  std::vector<Eigen::Triplet<double>> tripletList;
  unsigned int iterator = p;

  for( unsigned int j = 0; j < r; j++){
    for( unsigned int i = 0; i < r; i++){

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

        // build parameter label if needed
        // std::string par_lab = par_type_lab + "_" + std::to_string(node_i) + std::to_string(node_j);
        // if( par_type == 1){
        //   par_lab += "(" + std::to_string(cat_node_i) + ")";
        // } else if(par_type == 2){
        //   if(node_i == node_j & j!=i){
        //     par_lab = "Non relevant";
        //   }else{
        //     par_lab += "(" + std::to_string(cat_node_i) + ","+ std::to_string(cat_node_j) +")";
        //     }
        // }

        // plug-in vector elements in the matrix
        if(!(par_type == 2 & node_i == node_j & j!=i)){
          //edgesMat(i, j) = edgesMat(j, i) = theta(iterator);

          tripletList.push_back(Eigen::Triplet<double>(i, j, theta(iterator)));
          iterator ++;
        }

        // Rcpp::Rcout <<"("<<i<<","<<j<<"), " <<"Node i:" << node_i << ", type "<< node_i_type << ". Node j:" << node_j<< ", type "<< node_j_type;
        // Rcpp::Rcout << " ----> par: " << par_lab << "\n";
      }
    }
  }
  Eigen::SparseMatrix<double> edgesMat(r,r);
  edgesMat.setFromTriplets(tripletList.begin(), tripletList.end());
  return(edgesMat);

}

//' @export
// [[Rcpp::export]]
const Eigen::VectorXd theta_to_nodesPar(
    const Eigen::VectorXd &theta,
    const std::vector<unsigned int> nodes_type,
    const unsigned int p
){

  return theta.segment(0, p);
}

//' @export
// [[Rcpp::export]]
double compute_scale(
  const unsigned int sampling_scheme,
  const unsigned int n,
  const unsigned int n_nodes,
  const double prob,
  const unsigned int m,
  const unsigned int batch
){

  double output;
  switch(sampling_scheme){
  case 0:
    output = 1/double(n);
    break;
  case 1:
    output = 1/(double(n)*prob);
    break;
  case 2:
    output = 1/double(batch);
    break;
  case 3:
    output = n_nodes/double(m*batch);
    break;
  }

  return output;
}

//'@export
// [[Rcpp::export]]
Rcpp::NumericMatrix rmultinom_wrapper(const double prob, const unsigned int classes, const unsigned int batch, const unsigned int K) {

  Rcpp::NumericVector probs(classes, prob);
  Rcpp::IntegerVector outcome(classes);
  R::rmultinom(batch, probs.begin(), classes, outcome.begin());


  Rcpp::NumericMatrix out(classes, K);
  for(unsigned int j = 0; j < K; j++){
    out(Rcpp::_,j) = outcome;
  }

  return out;
}
#endif
