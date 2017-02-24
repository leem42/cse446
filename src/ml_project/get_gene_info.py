import mygene
import pprint
import uniprot as uni

mg = mygene.MyGeneInfo()

xli = ['ZNF502','TSPY3','PRPSAP2','UHRF1BP1','CHIT1','LOC728392']
out = mg.querymany(xli, scopes='symbol', fields='uniprot', species='human')
pprint.pprint(out)

print
print

#print uni.map('P31749', f='ACC', t='P_ENTREZGENEID') # map single id
#print uni.map(['P31749','Q16204'], f='ACC', t='P_ENTREZGENEID') # map list of ids
print uni.retrieve('P31749')
#print uni.retrieve(['P31749','Q16204'])
