#download data
mkdir data
cd data

#get wordsim353
wget http://www.cs.technion.ac.il/~gabr/resources/data/wordsim353/wordsim353.zip
unzip wordsim353.zip -d wordsim
mv wordsim/combined.tab wordsim353.txt
rm -Rf wordsim
rm wordsim353.zip

#get wordsim-sim and wordsim-rel
wget http://alfonseca.org/pubs/ws353simrel.tar.gz
tar -xvf ws353simrel.tar.gz
mv wordsim353_sim_rel/wordsim_relatedness_goldstandard.txt wordsim-rel.txt
mv wordsim353_sim_rel/wordsim_similarity_goldstandard.txt wordsim-sim.txt
rm -Rf wordsim353_sim_rel
rm ws353simrel.tar.gz

#get simlex999
wget http://www.cl.cam.ac.uk/~fh295/SimLex-999.zip
unzip SimLex-999.zip
awk -v OFS='\t' '{print $1,$2,$4}' SimLex-999/SimLex-999.txt > SimLex-999.txt
rm -Rf SimLex-999
rm SimLex-999.zip

#download pre-processed PPDB XL training data, ppdb datasets, n-gram feature sets, etc.
wget http://ttic.uchicago.edu/~wieting/charagram-demo.zip
unzip -j charagram-demo.zip
rm charagram-demo.zip