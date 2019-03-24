from math import ceil
a = [i for i in range(50)]
b = [i for i in range(50)]

for i in range(5,6):
    for j in range(0,1):
        for k in range(0,1):
            window = i
            #print(window)
            index = j
            #print(index)
            extra = k
            #print(extra)
            ds = 5
            offset = (index-window) % ds
            ri = (index+offset)//ds
            l = len(range((ds-offset)%ds,extra+window,ds))
            li = ri+l
            sub_a = a[ri:li]
            #sub_a = a[(index+offset)//ds:ceil((index+extra+window+phase)/ds)] 
            c = ["x"]*window+["_"]*50
            #print(sub_a)
            c[(ds-offset)%ds+index:index+extra+window:ds] = "*"*len(c[(ds-offset)%ds+index:index+extra+window:ds])
            print(c)
            d = c[index:index+extra+window]
            #i give up
            c_ = [x for x in c]
            c[index:index+extra+window] = "*"*len(c[index:index+extra+window])
            print(c)
            try:
                d[(ds-offset)%ds::ds] = sub_a
            except Exception:
                print(window)
                print(index)
                print(extra)





