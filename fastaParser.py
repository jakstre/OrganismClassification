


#TODO maybe edit parsing
def readNames(fasta):
    name, seq = None, []
    for line in fasta:
        line = line.rstrip()
        if line.startswith(">"):
            _,_, s = line.partition(" ")
            s,_,_ = s.partition(";")
            parts = s.split()
            if len(parts)>2:
                i = 0
                tmp = ""
                #while i < len(parts) and parts[i] != "sp."  and parts[i]!="Lineage=Root":
                while i < len(parts)  and parts[i]!="Lineage=Root": 
                    if tmp=="":
                        tmp = parts[i]
                    else:
                        tmp+=" "+ parts[i]
                    i+=1
                s = tmp
            if name: 
                yield (name, ''.join(seq))
            name, seq = s, []
        else:
            seq.append(line)
    if name: 
        yield (name, ''.join(seq))


# return dictionary: key = species name, value = list of DNA sequences (not unique)
def getNamesSeqLists(fasta):
    namesSeqMap = {}
    for name, seq in readNames(fasta):
        if name in namesSeqMap:
            namesSeqMap[name].append(seq)
        else:
            namesSeqMap[name] = [seq]
    return namesSeqMap


# return dictionary: key = species name, value = set of DNA sequences
def getNamesSeqSet(fasta):
    namesSeqMap = {}
    for name, seq in readNames(fasta):
        if name in namesSeqMap:
            namesSeqMap[name].add(seq)
        else:
            namesSeqMap[name] = set([seq])
    return namesSeqMap 
