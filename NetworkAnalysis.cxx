/* std classes */
#include <iostream>
#include <string>
#include <vector>
#include <sstream> // to convert int to std::string
#include <math.h> // for the sqrt
#include <fstream> // to open a file

/* GenerateCLP */
#include "NetworkAnalysisCLP.h" //generated when ccmake

  /////////////////////////////////////////
 //               MATRIX                //
/////////////////////////////////////////

std::vector< std::vector< int > > Get2DMatrix( std::vector< int > M )
{
	int Size = (int)sqrt( M.size() );

	std::vector< std::vector< int > > M2D;
	M2D.resize(Size);

	for(int i=0;i<Size;i++)
	{
		M2D[i].resize(Size);

		for(int j=0;j<Size;j++) M2D[i][j]=M[ i*Size + j ];
	}

	return M2D;
}

void DisplayMatrix( std::vector< std::vector< int > > M )
{
	std::cout<<"|"<<std::endl<<"| Connectivity Matrix:"<<std::endl;

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		std::cout<<"| ";

		for(unsigned int j=0 ; j<M.size() ; j++)
		{
	//		std::cout<< M[i][j] << " ";
/* To display ones and spaces for 0s:*/
			if(M[i][j]==0) std::cout<< "  ";
			else std::cout<< "1 ";

		}

		std::cout<<std::endl;
	}
}

std::vector< std::vector< int > > RandomMatrix(int size)
{
	std::vector< std::vector< int > > M;

	M.resize(size);

	for(int i=0;i<size;i++)
	{
		M[i].resize(size);

		for(int j=0;j<size;j++)
		{
			if( rand()%10 > 3 ) M[i][j] = 0; // 70% to have no link, 30% to have a link
			else M[i][j] = 1;
		}
	}
	return M;
}

std::vector< int > OpenMatrixFile(std::string path) // returns the matrix as a 1D vector -> then use Get2DMatrix() // returns empty vector if unopenable file
{
	std::vector< int > M;
	std::ifstream file (path.c_str() , std::ios::in );  // opening in reading
	if(! file) // error while opening
	{
		std::cout<<"| File can not be opened"<<std::endl;
		return M; // returns empty vector
	}

	std::string line;
	getline(file, line);
	int value;
	for(unsigned int j=0; j<line.size()/2 +1 ;j++) // half characters are comma, so /2
	{
		std::istringstream ( (&line[2*j]) ) >> value;
		M.push_back(value); 
	}

	file.close();

	return M;
}

  /////////////////////////////////////////
 //           BASIC MEASURES            //
/////////////////////////////////////////

int GetNbLinks( std::vector< std::vector< int > > M )
{
	int nbLinks = 0;

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		for(unsigned int j=0 ; j<M.size() ; j++)
		{
			nbLinks = nbLinks + M[i][j]; // /2 if binary
		}
	}

	return nbLinks;
}

int GetDegree( std::vector< std::vector< int > > M , int Node ) // ki // Node = i
{
	int Degree = 0;

	for(unsigned int j=0 ; j<M.size() ; j++) Degree = Degree + M[Node][j];

	return Degree;
}

double GetMeanDegree( std::vector< std::vector< int > > M )
{
	double Mean=0;

	for(unsigned int i=0 ; i<M.size() ; i++) Mean = Mean + GetDegree (M,i);

	Mean = Mean / M.size();

	return Mean;
}

void DisplayDegreeDistribution( std::vector< std::vector< int > > M ) // table + "graph"
{
	/* Get Degree values */
	std::vector< int > Degrees;
	Degrees.resize( M.size() ); // 1 degree for each node
	for(unsigned int i=0 ; i<M.size() ; i++) Degrees[i] = GetDegree (M,i);

	/* Compute Distribution table*/
	std::vector< std::vector< int > > Distrib; // 2 rows table
	Distrib.resize( 2 ); // 2 rows : one for the degree values, one for the frequencies for each value

	// Sort the values -> get Distrib[0]
	sort( Degrees.begin(), Degrees.end() );

	for(unsigned int i=0 ; i<Degrees.size() ; i++)
	{
		int OK=1;
		for(unsigned int j=0 ; j<Distrib[0].size() ; j++) if(Distrib[0][j]==Degrees[i]) OK=0; // test if the degree is already in Distrib

		if(OK==1) Distrib[0].push_back(Degrees[i]); // if it is not, push it
	}

	// Compute the frequency -> get Distrib[1]
	Distrib[1].resize( Distrib[0].size() );
	for(unsigned int i=0 ; i<Distrib[0].size() ; i++) // for all the different degrees
	{
		Distrib[1][i]=0; // frequency of the ith degree
		for(unsigned int j=0 ; j<Degrees.size() ; j++) if(Degrees[j]==Distrib[0][i]) Distrib[1][i]++;
	}

	/* Display Distribution Table */
	std::cout<<"| Distribution Table:" << std::endl;
	std::cout<<"| Degree | Frequency" << std::endl;
	for(unsigned int i=0 ; i<Distrib[0].size() ; i++)
	{
		if(Distrib[0][i]<10) std::cout<<"|      "<< Distrib[0][i] << " | " << Distrib[1][i] << std::endl;
		else if(Distrib[0][i]>=10 && Distrib[0][i]<100) std::cout<<"|     "<< Distrib[0][i] << " | " << Distrib[1][i] << std::endl;
		else if(Distrib[0][i]>=100) std::cout<<"|    "<< Distrib[0][i] << " | " << Distrib[1][i] << std::endl;
	}

	/* Distribution Graph */
	std::cout<<"| Distribution Graph: " << std::endl;
	int MaxFreq=0;
	for(unsigned int i=0 ; i<Distrib[0].size() ; i++) if(Distrib[1][i]>MaxFreq) MaxFreq = Distrib[1][i];

	for(unsigned int i=MaxFreq ; i>0; i--)
	{
		std::cout<<"| ";
		for(unsigned int j=0 ; j<Distrib[0].size() ; j++) // for all the values
		{
			if(Distrib[1][j] >= (int)i)
			{
				if(Distrib[0][j]<10) std::cout<<"- ";
				else if(Distrib[0][j]>=10 && Distrib[0][j]<100) std::cout<<"-  ";
				else if(Distrib[0][j]>=100) std::cout<<" -  ";
			}
			else
			{
				if(Distrib[0][j]<10) std::cout<<"  ";
				else if(Distrib[0][j]>=10 && Distrib[0][j]<100) std::cout<<"   ";
				else if(Distrib[0][j]>=100) std::cout<<"    ";
			}
		}
		std::cout<< std::endl;
	}
	std::cout<<"| ";
	for(unsigned int i=0 ; i<Distrib[0].size() ; i++) std::cout<< Distrib[0][i]<< " ";// display the values at the bottom of the graph
	std::cout<< std::endl;
}

int GetShortestPathLength( std::vector< std::vector< int > > M, int source, int target) // basis for measuring integration // dij
{
/*  Wikipedia : http://en.wikipedia.org/wiki/Dijkstra%27s_algorithm for explanation and algorithm  */

	std::vector< int > dist; // contains the distance sum for each node
	std::vector< int > previous; // contains the previous node for each node
	std::vector< int > Q; // contains the nodes to study

	for(unsigned int i=0 ; i<M.size() ; i++)
	{ 
		dist.push_back(10000); // fill the dist table with "infinity"
		previous.push_back(-1); // set the previous table to "undefined"
		Q.push_back(i);
	}

	dist[source]=0;

	int u, alt;
	while(Q.size()!=0)
	{
		int minDist=100000;

		for(unsigned int i=0 ; i<Q.size() ; i++)
		{
			if( dist[Q[i]] < minDist)
			{
				minDist=dist[Q[i]];
				u=Q[i]; // we study the node in Q with the minimum dist
			}
		}

		Q.erase( std::remove( Q.begin(), Q.end(), u ) , Q.end() );

		if(u==target) break; // Now we can read the shortest path from source to target by iteration: see below
		if(dist[u]==10000) return 0; // all remaining vertices are inaccessible from source (their dist is 10000 so has not been changed): the target too, so return 0 because SOURCE AND TARGET ARE NOT CONNECTED.

		for(unsigned int i=0 ; i<M.size() ; i++)
		{
			if(M[u][i]==1 || M[i][u]==1) // if neighbor of u
			{
				alt = dist[u] + 1; // alt: length of the path from the root node to the neighbor node v if it were to go through u // 1 = distance between u and v
				if(alt<dist[i])
				{
					dist[i]=alt;
					previous[i]=u;
				}
			}
		}
	} // while(Q.size()!=0)

	std::vector< int > sequence; // sequence contains the indexes of the nodes in the shortest path
	u=target;
	
	while(previous[u]!=-1) // by the way, if target = source, then previous[u] = previous[target] = previous[source]=-1 => do not enter the while loop -> returns 0
	{
		sequence.push_back(u);
		u=previous[u];
	}

	return sequence.size(); // the size of "sequence" is the number of links to go to the target, so the distance.
}

int GetNumberOfTriangles( std::vector< std::vector< int > > M , int Node ) // basis for measuring segregation // ti // Node = i
{
	int nbOfTriangles = 0;

	for(unsigned int j=0 ; j<M.size() ; j++)
	{
		for(unsigned int h=0 ; h<M.size() ; h++) nbOfTriangles = nbOfTriangles + ( M[Node][j] * M[Node][h] * M[j][h] );
	}

	nbOfTriangles = nbOfTriangles/2;

	return nbOfTriangles;
}

  /////////////////////////////////////////
 //      MEASURES OF INTERGRATION       //
/////////////////////////////////////////

double GetCharacteristicPathLength( std::vector< std::vector< int > > M ) // L
{
	double L = 0;
	double Li; // average distance between node i and all other nodes

	int SPL; // Shortest Path Length : dij

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		Li=0;

		for(unsigned int j=0 ; j<M.size() ; j++)
		{
			if(j!=i)
			{
				SPL = GetShortestPathLength(M,i,j);
				Li = Li + SPL;
			}
		}

		Li = Li / ( M.size()-1 ); // n= M.size()
		L = L + Li ;
	}

	L = L * 1/M.size(); // n= M.size()

	return L;
}

double GetGlobalEfficiency( std::vector< std::vector< int > > M ) // E
{
	double E = 0;
	double Ei; // efficiency of node i

	int SPL; // Shortest Path Length : dij

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		Ei=0;

		for(unsigned int j=0 ; j<M.size() ; j++)
		{
			if(j!=i)
			{
				SPL = GetShortestPathLength(M,i,j);
				if(SPL!=0) Ei = Ei + 1/SPL; // GetShortestPathLength returns 0 if the source and target are not connected
			}
		}

		Ei = Ei / ( M.size()-1 ); // n= M.size()
		E = E + Ei ;
	}

	E = E * 1/M.size(); // n= M.size()

	return E;
}

  /////////////////////////////////////////
 //       MEASURES OF SEGREGATION       //
/////////////////////////////////////////

double GetClusteringCoefficient( std::vector< std::vector< int > > M ) // C
{
	double C = 0;

	int degree, nbTriangles; // ki, ti
	double Ci;

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		degree = GetDegree(M,i);
		if( degree > 2 ) // Ci=0 for ki<2
		{
			nbTriangles = GetNumberOfTriangles(M,i);
			Ci = 2*nbTriangles/(double)( degree*(degree-1) ) ; // clustering coefficient of node i
			C = C + Ci ;
		}
	}

	C = C * 1/M.size(); // n= M.size()

	return C;
}

double GetTransivity( std::vector< std::vector< int > > M ) // T
{
	double T;

	double Tnum=0;
	double Tdenom=0;

	int degree; // ki

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		Tnum = Tnum + 2* GetNumberOfTriangles(M,i);
		degree =  GetDegree(M,i);
		Tdenom = Tdenom + ( degree*(degree-1) );
	}

	T = Tnum / Tdenom ;

	return T;
}

double GetLocalEfficiency( std::vector< std::vector< int > > M ) // Eloc
{
	double Eloc = 0;
	double Eloci; // local efficiency of node i

	int SPL, degree; // Shortest Path Length : dij // Degree : ki

	int OKj=0;
	int OKh=0;
	std::vector< std::vector< int > > MNeighborsOfi; // "sub Matrix" that will contain only the interesting nodes: i, neighbors of i, j and h => "sub network"
	std::vector< int > NodesToKeep; // at each will iteration, will contain the numbers of the interesting nodes for the creation of the sub matrix
	int indexj, indexh=0; // indexes of the source and target in the sub matrix

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		Eloci=0;

		degree =  GetDegree(M,i);
		if( degree>2 ) // Eloci=0 for ki<2
		{

			for(unsigned int j=0 ; j<M.size() ; j++)
			{
				if(j!=i)
				{
					for(unsigned int h=0 ; h<M.size() ; h++)
					{
						// testing if j and h are accessible from neighbors of i = if they touch one of them
						OKj=0;
						OKh=0;
						for(unsigned int k=0 ; k<M.size() ; k++)
						{
							if( M[k][i]==1 || M[i][k]==1 ) // if( k is a neighbor of i)
							{
								if( k==j || M[k][j]==1 || M[j][k]==1 ) // if ( k touch j or is j) (if j is i, j touches a neighbor of i)
								{
									OKj=1;
								}

								if( k==h || M[k][h]==1 || M[h][k]==1 ) // if ( k touch h or is h) (if h is i, h touches a neighbor of i)
								{
									OKh=1;
								}
							}
						}

						if(j!=h && OKj==1 && OKh==1) // if j==h, SPL=0 so NO 1/SPL => take into account the path joining one to another ? NO 
						{				// computes only if source AND target touch a neighbor or are a neighbor
							// Creation of a "sub Matrix" that will contain only the interesting nodes: i, neighbors of i, j and h
							// for all nodes k in M : k is to keep if : k is a neighbor of i AND k !=i,j,h :
							NodesToKeep.clear();
							for(unsigned int k=0 ; k<M.size() ; k++) if( (M[i][k]==1 || M[k][i]==1) && k!=i && k!=j && k!=h ) NodesToKeep.push_back(k);
							NodesToKeep.push_back(i);
							if(j==i) indexj=NodesToKeep.size()-1;
							if(h==i) indexh=NodesToKeep.size()-1;
							if(j!=i)
							{
								NodesToKeep.push_back(j);
								indexj=NodesToKeep.size()-1;
							}
							if(h!=i && h!=j)
							{
								NodesToKeep.push_back(h);
								indexh=NodesToKeep.size()-1;
							}

							// Creation of the "sub Matrix"
							MNeighborsOfi.clear();
							MNeighborsOfi.resize( NodesToKeep.size() );
							for(unsigned int k=0 ; k<MNeighborsOfi.size() ; k++)
							{
								MNeighborsOfi[k].resize( NodesToKeep.size() );

								for(unsigned int l=0 ; l<MNeighborsOfi.size() ; l++) MNeighborsOfi[k][l]=M[ NodesToKeep[k] ][ NodesToKeep[l] ];
							}

							// Compute the SPL with the sub matrix
							SPL = GetShortestPathLength(MNeighborsOfi,indexj,indexh); // shortest path between j and h, that contains only neighbors of i
							if(SPL!=0) Eloci = Eloci + M[i][j]*M[i][h]/SPL; // GetShortestPathLength returns 0 if the source and target are not connected
						} // if(j!=h && OKj==1 && OKh==1)
					} // for(unsigned int h=0 ; h<M.size() ; h++)
				} // if(j!=i)
			} // for(unsigned int j=0 ; j<M.size() ; j++)

			Eloci = Eloci / ( degree*(degree-1) ); // n= M.size()
			Eloc = Eloc + Eloci ;

		} // if( degree>2 )
	}

	Eloc = Eloc * 1/M.size(); // n= M.size()

	return Eloc;
}

  /////////////////////////////////////////
 //       MEASURES OF RESILIENCE        //
/////////////////////////////////////////

double GetAssortativityCoefficient( std::vector< std::vector< int > > M ) // r
{
	double r=0; // so the return value in 0 if problem
	double rnum1 = 0; // before the '-'
	double rnum2 = 0; // after the '-'
	double rdenom1 = 0; // before the '-'
	double rdenom2 = 0; // after the '-'

	int degreei, degreej; // ki, kj

	for(unsigned int i=0 ; i<M.size() ; i++)
	{
		degreei = GetDegree(M,i);
		for(unsigned int j=0 ; j<M.size() ; j++)
		{
			degreej = GetDegree(M,j);

			rnum1 = rnum1 + (degreei * degreej);
			rnum2 = rnum2 + 0.5*(degreei + degreej);
			rdenom1 = rdenom1 + 0.5*(degreei*degreei + degreej*degreej);
			rdenom2 = rdenom2 + 0.5*(degreei + degreej);
		}
	}

	rnum1 = rnum1/M.size(); // l= M.size()
	rnum2 = rnum2/M.size() * rnum2/M.size(); // l= M.size()
	rdenom1 = rdenom1/M.size(); // l= M.size()
	rdenom2 = rdenom2/M.size() * rdenom2/M.size(); // l= M.size()

	if(rdenom1-rdenom2 != 0) r = (rnum1-rnum2) / (rdenom1-rdenom2);

	return r;
}

  /////////////////////////////////////////
 //            OTHER CONCEPTS           //
/////////////////////////////////////////

double GetSmallWorldness( std::vector< std::vector< int > > M ) // S
{
	double S,C,Crand,L,Lrand;
	std::vector< std::vector< int > > RandM = RandomMatrix( M.size() );

	C = GetClusteringCoefficient( M );
	Crand = GetClusteringCoefficient( RandM );
	L = GetCharacteristicPathLength( M );
	Lrand = GetCharacteristicPathLength( RandM );

	S = (C/Crand) / (L/Lrand);

	return S;
}

  /////////////////////////////////////////
 //           MAIN FUNCTION             //
/////////////////////////////////////////

int main (int argc, char *argv[])
{
	PARSE_ARGS; //thanks to this line, we can use the variables entered in command line as variables of the program
	//std::vector< int > Matrix, file ResultsFile, bool isWeighted, file MatrixFile

/* Input Test */
	if(Matrix.size()==0)
	{
		if( !MatrixFile.empty() ) Matrix=OpenMatrixFile( MatrixFile );
		else
		{
			std::cout<<"| Please give a Matrix for Analysis: Abort"<<std::endl;
			return -1;
		}
	}

/* Matrix Test */
	if(Matrix.size()==0)
	{
		std::cout<<"| Please give a non empty Matrix for Analysis: Abort"<<std::endl;
		return -1;
	}

	if(Matrix.size()==1) // because divide by n-1
	{
		std::cout<<"| Please give a Matrix and not a scalar: Abort"<<std::endl;
		return -1;
	}

	if ( ! fmod( sqrt( Matrix.size() ) , 1) == 0) // if sqrt not an integer, then matrix not square
	{
		std::cout<<"| Please give a Square Matrix : Abort"<<std::endl;
		return -1;
	}

/* Matrix */
	std::vector< std::vector< int > > M = Get2DMatrix( Matrix ); // now we can use M[][]

	int nbNodes=M.size();
	std::cout<<"|"<<std::endl<<"| Number of nodes in the network: "<< nbNodes <<std::endl;
	int nbLinks = GetNbLinks(M);
	std::cout<<"| Number of links in the network: "<< nbLinks <<std::endl;

	DisplayMatrix( M );

/* Degree Distribution */
	std::cout<<"|"<<std::endl<< "| => Degree Distribution: " <<std::endl;
	double MeanDegree = GetMeanDegree( M );
	std::cout<<"| Mean Degree (Density) = " << MeanDegree <<std::endl;

	DisplayDegreeDistribution( M );

/* Measures of integration */
	std::cout<<"|"<<std::endl<< "| => Measures of Integration: " <<std::endl;
	double CharacteristicPathLength = GetCharacteristicPathLength( M );
	std::cout<<"| Characteristic Path Length = " << CharacteristicPathLength <<std::endl;
	double GlobalEfficiency = GetGlobalEfficiency( M );
	std::cout<<"| Global Efficiency = " << GlobalEfficiency <<std::endl;

/* Measures of segregation */
	std::cout<<"|"<<std::endl<< "| => Measures of Segregation:" <<std::endl;
	double ClusteringCoefficient = GetClusteringCoefficient( M );
	std::cout<<"| Clustering Coefficient = " << ClusteringCoefficient <<std::endl;
	double Transivity = GetTransivity( M );
	std::cout<<"| Transivity = " << Transivity <<std::endl;
	double LocalEfficiency = GetLocalEfficiency( M );
	std::cout<<"| Local Efficiency = " << LocalEfficiency <<std::endl;

/* Measures of resilience */
	std::cout<<"|"<<std::endl<< "| => Measures of Resilience:" <<std::endl;
	double AssortativityCoefficient = GetAssortativityCoefficient( M );
	std::cout<<"| Assortativity Coefficient = " << AssortativityCoefficient <<std::endl;

/* Other Concepts */
	std::cout<<"|"<<std::endl<< "| => Other Concepts:" <<std::endl;
	double SmallWorldness = GetSmallWorldness( M );
	std::cout<<"| Small-Worldness = " << SmallWorldness <<std::endl;

/* Open and write the file */
	if(! ResultsFile.empty() )
	{
		std::ofstream file (ResultsFile.c_str() , std::ios::out | std::ios::trunc);  // opening in writing with erasing the open file
		if(! file) // error while opening
		{
			std::cout<<"| File can not be opened: Nothing will be saved"<<std::endl;
			return 0; // nothing will be saved
		}

		file << "Number of nodes in the network: " << nbNodes << std::endl;
		file << "Number of links in the network: " << nbLinks << std::endl;

		file <<std::endl<<"Connectivity Matrix:"<<std::endl;
		for(unsigned int i=0 ; i<M.size() ; i++)
		{
			for(unsigned int j=0 ; j<M.size() ; j++) file << M[i][j] << " ";
			file <<std::endl;
		}

		file <<std::endl<< "=> Degree Distribution: " <<std::endl;
		file <<"Mean Degree (Density) = " << MeanDegree <<std::endl;
		// DisplayDegreeDistribution( M ); ???

		file <<std::endl<< "=> Measures of Integration: " <<std::endl;
		file <<"Characteristic Path Length = " << CharacteristicPathLength <<std::endl;
		file <<"Global Efficiency = " << GlobalEfficiency <<std::endl;

		file <<std::endl<< "=> Measures of Segregation:" <<std::endl;
		file <<"Clustering Coefficient = " << ClusteringCoefficient <<std::endl;
		file <<"Transivity = " << Transivity <<std::endl;
		file <<"Local Efficiency = " << LocalEfficiency <<std::endl;

		file <<std::endl<< "=> Measures of Resilience:" <<std::endl;
		file <<"Assortativity Coefficient = " << AssortativityCoefficient <<std::endl;

		file <<std::endl<< "=> Other Concepts:" <<std::endl;
		file <<"Small-Worldness = " << SmallWorldness <<std::endl;

		file.close();
	} // if(! ResultsFile.empty() )

//	DisplayDegreeDistribution( RandomMatrix(500) ); // to test the degree distribution

/* End of Main function */

	return 0;
}

