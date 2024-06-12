#include "TMVA/Reader.h"
#include "TFile.h"
#include "TXMLEngine.h"

void evaluate(TMVA::Reader *tmva_reader, map<string, Float_t> & float_store, map<string, Int_t> & int_store, 
              vector<string> & tmva_variables, vector<string> & tmva_spectators, string tmva_output, string label,
              TString input_filename, TString input_treename, TString output_filename, bool append_root = false) {
  TFile *input = TFile::Open(input_filename);
  TTree* tree = (TTree*)input->Get(input_treename);
  for (auto var : tmva_variables) tree->SetBranchAddress( var.c_str(), &float_store[var] );
  for (auto var : tmva_spectators) {
    if (var != "sampleID") tree->SetBranchAddress( var.c_str(), &float_store[var] );
    else tree->SetBranchAddress( var.c_str(), &int_store[var] );
  }
  tree->SetBranchAddress( label.c_str(), &int_store[label] );

  TFile * output = 0;
  if (append_root) output = new TFile(output_filename, "update");
  else output = new TFile(output_filename, "recreate");
  TTree * out_tree = new TTree(input_treename,input_treename);
  for (auto var : tmva_variables) out_tree->Branch( var.c_str(), &float_store[var] );
  for (auto var : tmva_spectators) {
    if (var != "sampleID") out_tree->Branch( var.c_str(), &float_store[var] );
    else out_tree->Branch( var.c_str(), &int_store[var] );
  }
  out_tree->Branch( label.c_str(), &int_store[label] );
  out_tree->Branch( tmva_output.c_str(), &float_store[tmva_output] );

  // Event loop
  for (Long64_t ievt=0; ievt<tree->GetEntries();ievt++) {
    tree->GetEntry(ievt);
    float_store[tmva_output] = tmva_reader->EvaluateMVA(tmva_output);
    out_tree->Fill();
  }
  out_tree->Write();
  cout<<"Wrote to "<<output_filename<<endl;

  input->Close();
  output->Close();
}

XMLNodePointer_t getNChild(TXMLEngine & xml, XMLNodePointer_t node, int iChild=0) {
  XMLNodePointer_t child = xml.GetChild(node);
  for (int i = 0; i < iChild; i++) {
    child = xml.GetNext(child);
  }
  return child;
}

void parse_xml(string tmva_weights, vector<string> & tmva_variables, vector<string> & tmva_spectators) {
  TXMLEngine xml;
  XMLDocPointer_t xmldoc = xml.ParseFile(tmva_weights.c_str());
  XMLNodePointer_t mainnode = xml.DocGetRootElement(xmldoc);
  // Get variables
  XMLNodePointer_t xml_variables = getNChild(xml, mainnode, 2);
  int nVar = stoi(xml.GetAttr(xml_variables, "NVar"));
  for (int iVar = 0; iVar < nVar; ++iVar) {
    XMLNodePointer_t xml_variable = getNChild(xml, xml_variables, iVar);
    tmva_variables.push_back(xml.GetAttr(xml_variable, "Expression"));
  }
  // Get spectators
  XMLNodePointer_t xml_spectators = getNChild(xml, mainnode, 3);
  int nSpec = stoi(xml.GetAttr(xml_spectators, "NSpec"));
  for (int iSpec = 0; iSpec < nSpec; ++iSpec) {
    XMLNodePointer_t xml_spectator = getNChild(xml, xml_spectators, iSpec);
    tmva_spectators.push_back(xml.GetAttr(xml_spectator, "Expression"));
  }
  xml.FreeDoc(xmldoc);
  cout<<"Variables from xml: ";
  for (auto it : tmva_variables) cout<<it<<" ";
  cout<<endl;
  cout<<"Spectators from xml: ";
  for (auto it : tmva_spectators) cout<<it<<" ";
  cout<<endl;
}

using namespace TMVA;
void infer_tmva(string input_file, vector<string> trees, string mva_branch, string tmva_weights, string output_file) {

  // Load bdt tmva
  TMVA::Reader *tmva_reader = new TMVA::Reader( "!Color:!Silent" );

  string root_ntuple = input_file;
  string label = "classID";

  string output_root = output_file;
  //string tmva_weights;
  string tmva_output = mva_branch;

  // Parse xml file to get variables and spectators
  vector<string> tmva_variables;
  vector<string> tmva_spectators;
  parse_xml(tmva_weights, tmva_variables, tmva_spectators);

  // Make memory for variables
  map<string, Float_t> float_store;
  map<string, Int_t> int_store;
  for (auto var : tmva_variables) float_store[var];
  for (auto var : tmva_spectators) {
    if (var != "sampleID") float_store[var];
    else int_store[var];
  }
  float_store[tmva_output];
  int_store[label];

  // Add variables to reader
  for (auto var : tmva_variables) tmva_reader->AddVariable(var.c_str(), &float_store[var]);
  for (auto var : tmva_spectators) {
    if (var != "sampleID") tmva_reader->AddSpectator(var.c_str(), &float_store[var]);
    else tmva_reader->AddSpectator(var.c_str(), &int_store[var]);
  }
  tmva_reader->BookMVA(tmva_output.c_str(), tmva_weights.c_str());

  // Infer over trees
  int iTree = 0;
  for (auto tree_name : trees) {
    if (iTree == 0) evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, root_ntuple, tree_name, output_root);
    else evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, root_ntuple, tree_name, output_root, true);
    iTree++;
  }
  //evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, root_ntuple, train_tree, output_root);
  //evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, root_ntuple, eval_tree, output_root, true);
  //evaluate(tmva_reader, float_store, int_store, tmva_variables, tmva_spectators, tmva_output, label, root_ntuple, test_tree, output_root, true);

}

class infer_tmva_c {
  public:
    infer_tmva_c(string tmva_name, string tmva_weights) {
      m_tmva_reader = 0;
      m_tmva_name = tmva_name;
      m_tmva_weights = tmva_weights;

      // Load bdt tmva
      // Parse xml file to get variables and spectators
      parse_xml(tmva_weights, m_tmva_variables, m_tmva_spectators);
      // Make tmva_reader
      m_tmva_reader = new TMVA::Reader( "!Color:!Silent" );
      // Make memory for variables
      for (auto var : m_tmva_variables) {
        m_input_store[var];
        m_variable_names.push_back(var);
      }
      for (auto var : m_tmva_spectators) m_input_store[var];

      // Add variables to reader
      for (auto var : m_tmva_variables) m_tmva_reader->AddVariable(var.c_str(), &m_input_store[var]);
      for (auto var : m_tmva_spectators) m_tmva_reader->AddSpectator(var.c_str(), &m_input_store[var]);
      m_tmva_reader->BookMVA(m_tmva_name.c_str(), m_tmva_weights.c_str());
    }
    ~infer_tmva_c() {if (m_tmva_reader) delete m_tmva_reader;}
    // members
    TMVA::Reader * m_tmva_reader;
    string m_tmva_name;
    string m_tmva_weights;
    map<string, Float_t> m_input_store;
    vector<string> m_variable_names;
    vector<string> m_tmva_variables;
    vector<string> m_tmva_spectators;
    // methods
    vector<float> infer(vector<vector<double>> input_ntuple) {
      vector<float> results(input_ntuple.size());
      // Infer over input
      for (unsigned ievt=0; ievt<input_ntuple.size(); ievt++) {
        // Update input store
        for (unsigned ivar=0; ivar<m_tmva_variables.size(); ivar++)  {
          string variable_name = m_variable_names[ivar];
          m_input_store[variable_name] = input_ntuple[ievt][ivar];
        }
        // Infer
        results[ievt] = m_tmva_reader->EvaluateMVA(m_tmva_name);
      }
      return results;
    }
    float infer(vector<double> input_ntuple) {
      float result;
      // Update input store
      for (unsigned ivar=0; ivar<m_tmva_variables.size(); ivar++)  {
        string variable_name = m_variable_names[ivar];
        m_input_store[variable_name] = input_ntuple[ivar];
      }
      // Infer
      result = m_tmva_reader->EvaluateMVA(m_tmva_name);
      return result;
    }
};

// input_ntuple = [ [features values for event], next event, ...]
vector<float> infer_tmva_function(vector<vector<double>> input_ntuple, string tmva_name, string tmva_weights) {
//vector<float> infer_tmva_function(vector<vector<double>> input_ntuple) {

  vector<float> results(input_ntuple.size());
  string tmva_output = tmva_name;
  //string tmva_output = "BDT";
  //string tmva_weights = "tmva_bdt_hig19014_model/weights/TMVAClassification_BDT.weights.xml";


  // Parse xml file to get variables and spectators
  vector<string> tmva_variables;
  vector<string> tmva_spectators;
  parse_xml(tmva_weights, tmva_variables, tmva_spectators);

  // Load bdt tmva
  TMVA::Reader *tmva_reader = new TMVA::Reader( "!Color:!Silent" );

  // Make memory for variables
  map<string, Float_t> input_store;
  vector<string> variable_names;
  for (auto var : tmva_variables) {
    input_store[var];
    variable_names.push_back(var);
  }
  for (auto var : tmva_spectators) input_store[var];

  // Add variables to reader
  for (auto var : tmva_variables) tmva_reader->AddVariable(var.c_str(), &input_store[var]);
  for (auto var : tmva_spectators) tmva_reader->AddSpectator(var.c_str(), &input_store[var]);
  tmva_reader->BookMVA(tmva_output.c_str(), tmva_weights.c_str());

  // Infer over input
  for (unsigned ievt=0; ievt<input_ntuple.size(); ievt++) {
    // Update input store
    for (unsigned ivar=0; ivar<tmva_variables.size(); ivar++)  {
      string variable_name = variable_names[ivar];
      input_store[variable_name] = input_ntuple[ievt][ivar];
    }
    // Infer
    results[ievt] = tmva_reader->EvaluateMVA(tmva_output);
  }

  delete tmva_reader;

  return results;
}
