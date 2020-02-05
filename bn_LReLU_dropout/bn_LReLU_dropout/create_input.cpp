#include <string>
#include <sstream>
#include <iostream>
#include <fstream>
#include <math.h>
#include <algorithm>

using namespace std;

int GetLogBin( float* logscale, int nBins, float n )
{
  for( int i = 1; i < nBins; i++ ) {
    if( n < logscale[i] ) return i-1;
  }
  return nBins-1;
}

int GetPartNum( int* pl, int n_p, int p_id )
{
  for( int i = 0; i < n_p; i++ ) {
    if( pl[i] == p_id ) return i;
  }
  std::cout<<"Error! Wrong particle id!\n";
  return -1;
}

int main( int argc, const char* argv[] )
{
  //
  float p_max_inc(3.14), p_step_inc(3.14 / 20), Ninc(20), Nazm(20), p_max_azm(3.14), p_step_azm(6.28 / 20);
  int particle_list[28] = { -211, 111, 211, 2112, 2212, 311, 321, 221, 3122, -321,
			-311, 3212, 3222, 3112, 3322, 3312, 22, -2112, -2212, -3122,
			-3212, -3112, -3222, 3334, -3322, -3312, 333, -3334 };
  int particle_list1[28] = { -211, 111, 211, 2112, 2212, 311, 321, 221, 3122, -321,
  			-311, 3212, 3222, 3112, 3322, 3312, 22, -2112, -2212, -3122,
  			-3212, -3112, -3222, 3334, -3322, -3312, 333, -3334 };
  sort(&particle_list1[0], &particle_list1[28]);
  float p_logbin[20] = { 0.13229363,  0.28208885,  0.45170104,  0.64375183,  0.86120972,  1.1074359,
      1.38623623,  1.70192008,  2.05936688,  2.46410162,  2.92238018,  3.44128607,
      4.02883991,  4.69412337,  5.44741959,  6.3003721,   7.26616479,  8.3597257,
      9.59795775, 11. };
  //
  const int ii(28), jj(20), kk(20), ll(20);
  int matrix[ii][jj][kk][ll];
  //
  for( int iqgp = 0; iqgp < 2; iqgp++ ) {
    int n_event = 0;
    for( int ifl = 1; ifl < 6; ifl++ ) {
      string prefix_qgp = "nqgp/";
      if( iqgp ) prefix_qgp = "qgp/";
      string prefix_infile = "phsd50csr.auau.31.2gev.centr.0000";
      string prefix_outfile = "phsd50csr.auau.31.2gev.centr.";
      if( ifl > 9 ) prefix_infile = "phsd50csr.auau.31.2gev.centr.000";
      std::string prefix_nfile = std::to_string(ifl);
      ifstream ifile((prefix_qgp+prefix_infile+prefix_nfile+".dat").data());
      if(!ifile) return 1;
      for( int iev = 0; iev < 1000; iev++ ) {
        int nOfParticles, int_tmp;
        float float_tmp;
        string prefix_outdir = "dataset_new/";
        std::string prefix_nev = std::to_string(n_event);
        ifile >> nOfParticles >> int_tmp >> int_tmp >> float_tmp >> int_tmp;
        ifile >> int_tmp >> float_tmp >> float_tmp >> float_tmp >> float_tmp >> float_tmp >> float_tmp >> float_tmp >> float_tmp;
        for( int i = 0; i < ii; i++ )
          for( int j = 0; j < jj; j++ )
            for( int k = 0; k < kk; k++ )
              for( int l = 0; l < ll; l++ )
        	matrix[i][j][k][l] = 0;
        //
        const int max_len = 10;
        float line[max_len];
        const char ch = '\n';
        char l1[1000];
        ifile.getline(l1, 0, ch);	// skip "\n" from the previous line
        for( int iP = 0; iP < nOfParticles; iP++ ) {
          ifile.getline(l1, 999,ch);
          string l1s1(l1);
          stringstream stream(l1s1);
          int real_len = max_len;
          for( int i = 0; i < real_len; i++ ) {
            stream >> line[i];
            if(!stream) {real_len = i; break;}
          }
          bool get_particle = binary_search(&particle_list1[0], &particle_list1[28], int(line[0]));
          if(!get_particle) continue;
          if( real_len < 8 ) { std::cout<<"Error! Array sizr is wrong! real_len = "<<real_len<<"\n"; continue; }
          float p_abs_val = sqrt(line[2]*line[2] + line[3]*line[3]  + line[4]*line[4]);
          float p_azm = atan2(line[3], line[2]);
          float p_inc = acos(line[4] / p_abs_val);
//          std::cout<<"> part: "<<iP<<";   a1: "<<line[1]<<";   a2: "<<line[2]<<";   a3: "<<line[3]<<";   a4: "<<line[4]<<"\n";
//          std::cout<<" --- p_abs_val: "<<p_abs_val<<";   p_azm: "<<p_azm<<";   p_inc: "<<p_inc<<"\n";
//          std::cout<<" --- p_step_azm: "<<p_step_azm<<"\n";

          int p_id = GetPartNum( particle_list, 28, int(line[0]) );
          int p_ind = GetLogBin( p_logbin, 20, p_abs_val ) + 1;
//          int p_azm_ind = int((p_azm + 1.02018) / p_step_azm);
          int p_azm_ind = int((p_azm + 3.14) / p_step_azm);
          int p_inc_ind = int(p_inc / p_step_inc);
//          std::cout<<" --- p_id: "<<p_id<<"\n";
//          std::cout<<" --- p_ind: "<<p_ind<<";   p_azm_ind: "<<p_azm_ind<<";   p_inc_ind: "<<p_inc_ind<<"\n";
          matrix[p_id][p_ind][p_azm_ind][p_inc_ind]++;
        }
        //
//        int iii = 0;
//        for( int i = 0; i < ii; i++ )
//          for( int j = 0; j < jj; j++ )
//            for( int k = 0; k < kk; k++ )
//              for( int l = 0; l < ll; l++ )
//        	if(matrix[i][j][k][l]) iii++;
//        std::cout<<" --- iii: "<<iii<<"\n";
//        for( int i = 0; i < ii; i++ ) {
//          for( int j = 0; j < jj; j++ )
//            for( int k = 0; k < kk; k++ )
//              for( int l = 0; l < ll; l++ ) {
////        	if(matrix[i][j][k][l]) std::cout<<"i: "<<i<<";   j: "<<j<<";   k: "<<k<<";   l: "<<l<<"\n";
////        	if(matrix[i][j][k][l]) std::cout<<matrix[i][j][k][l]<<"\n";
//              }
//        }
        ofstream jfile((prefix_outdir+prefix_qgp+prefix_infile+prefix_outfile+prefix_nev+"_event.dat").data(),std::ios::out|std::ios::app);
        for( int i = 0; i < ii; i++ ) {
          for( int j = 0; j < jj; j++ )
            for( int k = 0; k < kk; k++ )
              for( int l = 0; l < ll; l++ )
        	jfile << matrix[i][j][k][l] << " ";
          jfile << "\n";
        }
        jfile.close();
        n_event++;
      }
      ifile.close();
    }
  }
}
