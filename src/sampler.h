#ifndef sampler_H
#define sampler_H

// [[Rcpp::export]]
const unsigned int intMultinom(Eigen::VectorXd E_probs, bool verboseFLAG = false, const double tol = 1e-5){

  unsigned int resp = 100;
  if(1-E_probs.sum() > tol){
    Rcpp::Rcout << "Probs do not sum up to 1!\n";
  }else{
    unsigned int cat =  E_probs.size();
    Rcpp::NumericVector probs = Rcpp::wrap(E_probs);
    Rcpp::IntegerVector respVec(cat);

    R::rmultinom(1, probs.begin(), cat , respVec.begin());

    if(verboseFLAG )Rcpp::Rcout << respVec << "\n";

    resp = 0;
    for( unsigned int s = 0; s < cat; s++) {
      if(respVec(s) == 1) break;
      resp++;

    }
  }


  return resp;
}

// [[Rcpp::export]]
const unsigned int drawCatNode(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const unsigned int &node,
    const unsigned int &p,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    const bool verboseFLAG = false
){

  const unsigned int cat_node = nodes_type[node];
  Eigen::VectorXd probs = Eigen::VectorXd::Zero(cat_node);

  for(unsigned int l = 0; l < cat_node; l++){
    probs(l) = exp(eta_categorical_node(data, edgesPar, node, p, l, nodes_type, cumsum_nodes_type, verboseFLAG));
  }

  const double normConst = probs.sum();
  probs = probs.array()/normConst;
  if(verboseFLAG)Rcpp::Rcout << "probs:"; if(verboseFLAG)print_vec(probs);
  return intMultinom(probs, verboseFLAG);
}

// [[Rcpp::export]]
const double drawContNode(
    const Eigen::VectorXd &data,
    const Eigen::MatrixXd &edgesPar,
    const Eigen::VectorXd &nodesPar,
    const unsigned int &node,
    const unsigned int &p,
    const std::vector<unsigned int> &nodes_type,
    const std::vector<unsigned int> &cumsum_nodes_type,
    const bool verboseFLAG = false
){

  const double E = E_continuous_node(data, edgesPar, nodesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG);
  const double sd = sqrt(1/edgesPar(node, node));


  if(verboseFLAG)Rcpp::Rcout << "mean:"<< E << ", sd:"<<sd<<"\n";
  return R::rnorm(E, sd);
}

// [[Rcpp::export]]
Rcpp::List graphGibbsSampler(
    const Eigen::MatrixXd &edgesPar,
    const Eigen::VectorXd &nodesPar,
    const std::vector<unsigned int> nodes_type,
    const unsigned int warmup,
    unsigned int maxiter,
    const unsigned int m,
    const unsigned int skip,
    const bool verboseFLAG = false,
    const bool store_warmupFLAG = false
){
  maxiter += warmup;
  // Identify dimensions
  const unsigned int n_nodes = nodes_type.size();                    // number of nodes
  unsigned int p = 0;                                                // number of continuous nodes
  for( unsigned int node = 0; node < nodes_type.size(); node++){
    if(nodes_type[node] == 1) p++; else break;
  }
  const unsigned int q = n_nodes - p;                                // number of categorical nodes
  const unsigned int r = std::accumulate(nodes_type.begin(), nodes_type.end(), decltype(nodes_type)::value_type(0)); // dimension of edges parameter matrix

  // cumultative sum of nodes_type. Used to identify parameters in edgesPar
  std::vector<unsigned int> cumsum_nodes_type = nodes_type;
  for( unsigned int node = 1; node < n_nodes; node++) cumsum_nodes_type[node] = cumsum_nodes_type[node-1] + cumsum_nodes_type[node];


  std::vector<Eigen::MatrixXd> chains(m);

  for(unsigned int chain = 0; chain < m; chain ++){
    // initialize data vector
    Eigen::VectorXd dataVec = Eigen::VectorXd::Zero(n_nodes);
    for(unsigned int node = 0; node < n_nodes; node++){
      if(node < p){
        const double sd = sqrt(edgesPar(node, node));
        const double E = R::rnorm(0, 10);
        dataVec(node) = R::rnorm(E, sd);
      }else{
        int node_cat = nodes_type[node];
        //double prob = 1/double(node_cat);
        //Rcpp::Rcout << " node_type:"<< node_cat<< ", prob:" << prob << "\n";
        Eigen::VectorXd probs(node_cat);
        for(unsigned int cat = 0; cat < node_cat; cat ++) probs(cat =  R::rnorm(0, 10));
        probs = probs.array()/ probs.sum();
        //probs.fill(prob); //print_vec(probs);
        dataVec(node) = intMultinom(probs, verboseFLAG);
      }
    }

    Eigen::MatrixXd chain_data = Eigen::MatrixXd::Zero(maxiter+1, n_nodes); chain_data.row(0) = dataVec;



    /* GIBBS SAMPLING */

    unsigned int row = 1;
    unsigned int skip_counter = 0;
    for(unsigned int i = 0; i < maxiter; i ++){
      for(unsigned int node = 0; node < n_nodes; node++){
        if(node < p){
          dataVec(node) = drawContNode(dataVec, edgesPar, nodesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG);
        }else{
          dataVec(node) = drawCatNode(dataVec, edgesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG);
        }
      }

      if(i>=warmup){
        if(skip_counter == skip){
          chain_data.row(row) = dataVec;
          row++;
          skip_counter = 0;
        }else{
          skip_counter++;
        }
      } else if(store_warmupFLAG){
          chain_data.row(row) = dataVec;
          row++;
      }
    }

    chain_data.conservativeResize(row, n_nodes);
    chains[chain] = chain_data;
    //Eigen::MatrixXd data = chain_data.block(maxiter-n + 1, 0, n, n_nodes);
  }







  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("q") = q,
      Rcpp::Named("chains") = chains
      //Rcpp::Named("chain_data") = chain_data,
      //Rcpp::Named("data") = data
    );

  return output;
}

// [[Rcpp::export]]
Eigen::VectorXd cont_gamma(
  const Eigen::MatrixXd &edgesPar,
  const Eigen::VectorXd &nodesPar,
  const Eigen::VectorXd &cat_pattern,
  const unsigned int &p,
  const std::vector<unsigned int> &nodes_type,
  const std::vector<unsigned int> &cumsum_nodes_type,
  const bool verboseFLAG = false
){
  Eigen::VectorXd gamma = Eigen::VectorXd::Zero(p);
  for(unsigned int cont_node = 0; cont_node < p; cont_node ++){
    gamma(cont_node) += nodesPar(cont_node);
    for(unsigned int j = 0; j < cat_pattern.size(); j ++){
      const unsigned int cat_node = j + p;
      const unsigned int cat = cat_pattern(j);
      const unsigned int par_row = cumsum_nodes_type[cat_node-1]+cat;
      const double par = edgesPar(par_row, cont_node);
      gamma(cont_node) += par;
    }
  }

  return gamma;
}

//' @export
// [[Rcpp::export]]
Rcpp::List graphBlocksGibbsSampler(
    const Eigen::MatrixXd &edgesPar,
    const Eigen::VectorXd &nodesPar,
    const std::vector<unsigned int> nodes_type,
    const unsigned int warmup,
    unsigned int maxiter,
    const unsigned int m,
    const unsigned int skip,
    const bool verboseFLAG = false,
    const bool store_warmupFLAG = false
){
  maxiter += warmup;
  // Identify dimensions
  const unsigned int n_nodes = nodes_type.size();                    // number of nodes
  unsigned int p = 0;                                                // number of continuous nodes
  for( unsigned int node = 0; node < nodes_type.size(); node++){
    if(nodes_type[node] == 1) p++; else break;
  }
  const unsigned int q = n_nodes - p;                                // number of categorical nodes
  const unsigned int r = std::accumulate(nodes_type.begin(), nodes_type.end(), decltype(nodes_type)::value_type(0)); // dimension of edges parameter matrix

  // cumultative sum of nodes_type. Used to identify parameters in edgesPar
  std::vector<unsigned int> cumsum_nodes_type = nodes_type;
  for( unsigned int node = 1; node < n_nodes; node++) cumsum_nodes_type[node] = cumsum_nodes_type[node-1] + cumsum_nodes_type[node];

  // continuous conditional covariance
  Eigen::MatrixXd Ip = Eigen::MatrixXd::Identity(p,p);
  Eigen::MatrixXd B = (edgesPar.block(0,0,p,p)).selfadjointView<Eigen::Lower>();
  Eigen::LLT<Eigen::MatrixXd> llt(B);
  Eigen::MatrixXd Binv = llt.solve(Ip);
  Eigen::LLT<Eigen::MatrixXd> llt2(Binv);
  Eigen::MatrixXd C = llt2.matrixL();


  std::vector<Eigen::MatrixXd> chains(m);

  for(unsigned int chain = 0; chain < m; chain ++){
    // initialize data vector
    Eigen::VectorXd dataVec = Eigen::VectorXd::Zero(n_nodes);
    for(unsigned int node = 0; node < n_nodes; node++){
      if(node < p){
        const double sd = sqrt(edgesPar(node, node));
        const double E = R::rnorm(0, 10);
        dataVec(node) = R::rnorm(E, sd);
      }else{
        int node_cat = nodes_type[node];
        //double prob = 1/double(node_cat);
        //Rcpp::Rcout << " node_type:"<< node_cat<< ", prob:" << prob << "\n";
        Eigen::VectorXd probs(node_cat);
        for(unsigned int cat = 0; cat < node_cat; cat ++) probs(cat =  R::rnorm(0, 10));
        probs = probs.array()/ probs.sum();
        //probs.fill(prob); //print_vec(probs);
        dataVec(node) = intMultinom(probs, verboseFLAG);
      }
    }

    Eigen::MatrixXd chain_data = Eigen::MatrixXd::Zero(maxiter+1, n_nodes); chain_data.row(0) = dataVec;



    /* GIBBS SAMPLING */

    unsigned int row = 1;
    unsigned int skip_counter = 0;
    for(unsigned int i = 0; i < maxiter; i ++){

      // categorical nodes
      for(unsigned int node = p; node < n_nodes; node++){
          dataVec(node) = drawCatNode(dataVec, edgesPar, node, p, nodes_type, cumsum_nodes_type, verboseFLAG);
      }

      // continuous nodes
      Eigen::VectorXd z(p);
      for(unsigned int j = 0; j < p; j++) z(j) = R::rnorm(0,1);
      Eigen::VectorXd gamma = cont_gamma(edgesPar, nodesPar, dataVec.segment(p, q), p, nodes_type, cumsum_nodes_type, verboseFLAG);
      Eigen::VectorXd mean = Binv * gamma;
      if(i == 10000)   Rcpp::Rcout << mean << "\n";
      dataVec.segment(0, p) = mean + C*z;


      if(i>=warmup){
        if(skip_counter == skip){
          chain_data.row(row) = dataVec;
          row++;
          skip_counter = 0;
        }else{
          skip_counter++;
        }
      } else if(store_warmupFLAG){
        chain_data.row(row) = dataVec;
        row++;
      }
    }

    chain_data.conservativeResize(row, n_nodes);
    chains[chain] = chain_data;
    //Eigen::MatrixXd data = chain_data.block(maxiter-n + 1, 0, n, n_nodes);
  }







  Rcpp::List output =
    Rcpp::List::create(
      Rcpp::Named("p") = p,
      Rcpp::Named("q") = q,
      Rcpp::Named("Binv") = Binv,
      Rcpp::Named("chains") = chains
    //Rcpp::Named("chain_data") = chain_data,
    //Rcpp::Named("data") = data
    );

  return output;
}
#endif
