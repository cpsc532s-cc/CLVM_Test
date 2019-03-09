
a = [i for i in range(50)]
b = [i for i in range(50)]
window = 2
index = 0
extra=4
ds = 2
offset = (index-window) % ds
sub_a = a[index//ds:(index+extra+window)//ds] 
c = ["x"]*window+["_"]*50
print(sub_a)
print(offset)
print(ds+index-offset)
c[(ds-offset)%ds+index:index+extra+window:ds] = "*"*len(c[(ds-offset)%ds+index:index+extra+window:ds])
d = c[index:index+extra+window]
print(c)
c[index:index+extra+window] = "*"*len(c[index:index+extra+window])
print(c)
d[(ds-offset)%ds::ds] = sub_a
print(d)





