import numpy as np 
import pandas as pd 
# pd.options.mode.chained_assignment = None # to remove SettingWithCopyWarning

def ruleToText(rule, dimorder, dimvals):
    condition={"dimorder":dimorder,
                "dimvals":dimvals}

    catfun=catfuns[rule]
    stims = ['stim00',
        'stim01',
        'stim02',
        'stim03',
        'stim04',
        'stim05',
        'stim06',
        'stim07'
        ]
    allstims = [0,1,2,3,4,5,6,7]

    groupBool = [catfun(a) for a in range(8)]
    groupLabel=[]
    for ii in groupBool:
        if ii:
            groupLabel.append("B")
        else:
            groupLabel.append("A")

    stimcardspresented = []

    for ii in range(len(allstims)):
        prescard = allstims[ii]
        stim = stims[getstim(prescard,condition)]
        stimcardspresented.append(stim)

    #new march 21 2022
    label_code = {"size": ['large', 'small'],
                 "color": ['black','white'],
                 "shape": ['square','triangle']}

    labeled_stims = {"stim00": ['large','black','square'],
                 "stim01": ['large','white','square'],
                 "stim02": ['large','black','triangle'],
                 "stim03": ['large','white','triangle'],
                 "stim04": ['small','black','square'],
                 "stim05": ['small','white','square'],
                 "stim06": ['small','black','triangle'],
                 "stim07": ['small','white','triangle']
                }
    ruletext_templates = ['{feat1} cards are in Group A. {feat1inv} cards are in Group B.',
            '{feat1} {feat2} and {feat1inv} {feat2inv} cards are in Group A. {feat1} {feat2inv} and {feat1inv} {feat2} cards are in Group B.',
            '{feat1} cards are in Group A except for the {feat1exception} card. {feat1inv} cards are in Group B except for the {feat1invexception} card.',
            '{feat1} cards are in Group A except for the {feat1exception} card. {feat1inv} cards are in Group B except for the {feat1invexception} card.',
            '{feat1} cards are in Group A except for the {feat1exception} card. {feat1inv} cards are in Group B except for the {feat1invexception} card.',
            'Don\'t even try me yet!'
            ]

    carddata = { "png": stimcardspresented,
        "catBool": groupBool,
        "catLabel": groupLabel,
        "size": [labeled_stims[fname][0] for fname in stimcardspresented],
        "color": [labeled_stims[fname][1] for fname in stimcardspresented],
        "shape": [labeled_stims[fname][2] for fname in stimcardspresented],
        }
    df = pd.DataFrame(carddata)
    aCards = df.query("catLabel=='A'")
    bCards = df.query("catLabel=='B'")

    txt = ""

    if rule==0: # TYPE 1
        feat1=None
        feat1inv=None
        
        # One relevant feature dimension, let's find it
        df_a = aCards[['size','color','shape']]
        relevantDimension = df_a.columns[df_a.nunique() <= 1][0]
        feat1 = df_a.loc[0,relevantDimension]
        
        feat1inv = df[relevantDimension].unique()[df[relevantDimension].unique() != feat1][0]
        txt = ruletext_templates[rule].format(feat1=feat1.capitalize(), feat1inv=feat1inv.capitalize())

    elif rule==1: # TYPE 2
        # find two relevant dimensions
        df_a = pd.DataFrame() #aCards[['size','color','shape']]
        for col in ['size','color','shape']:
            df_a[col] = aCards[col].astype('category').cat.codes
        correlation = df_a.corr().abs()
        relcols = [col for col in correlation.columns if correlation[col].sum()==2]
        featurepairs = [list(key) for key in df_a.groupby(relcols).groups]
        # March 2022 FIX
        feat1 = label_code[relcols[0]][featurepairs[0][0]]
        feat2 = label_code[relcols[1]][featurepairs[0][1]]
        feat1inv = label_code[relcols[0]][featurepairs[1][0]]
        feat2inv = label_code[relcols[1]][featurepairs[1][1]]

        # old erroneous code
        # feat1 = aCards.reset_index().loc[featurepairs[0][0],relcols[0]]
        # feat2 = aCards.reset_index().loc[featurepairs[0][1],relcols[1]]
        # feat1inv = aCards.reset_index().loc[featurepairs[1][0],relcols[0]]
        # feat2inv = aCards.reset_index().loc[featurepairs[1][1],relcols[1]]
            
        txt = ruletext_templates[rule].format(feat1=feat1.capitalize(), feat2=feat2, feat1inv=feat1inv, feat2inv=feat2inv)
        
    elif rule>=2: # TYPE 3
        df_a = aCards[['size','color','shape']]
        relcols=[]
        featvals=[]
        for col in df_a.columns:
            cts = df_a[col].value_counts()
            for row in cts.iteritems():
                r=list(row)
                if r[1]>2:
                    relcols.append(col)
                    featvals.append(r[0])

        # multiple ways to explain the rule so choose one
        rndchoice = np.random.randint(0,len(relcols))
        relevantDimension = relcols[rndchoice]
        feat1 = featvals[rndchoice]
        feat1inv = df[relevantDimension].unique()[df[relevantDimension].unique() != feat1][0]

        # find row of B where relevantDimension==feat1
        df_b = bCards[['size','color','shape']]
        feat1exception = ", ".join(df_b[df_b[relevantDimension]==feat1].values[0])
        # find row of A where relevantDimension==feat1inv
        feat1invexception = ", ".join(df_a[df_a[relevantDimension]==feat1inv].values[0])
        
        txt = ruletext_templates[rule].format(feat1=feat1.capitalize(), feat1exception=feat1exception, feat1inv=feat1inv.capitalize(), feat1invexception=feat1invexception)
    elif rule>=5 | rule<0:
        print("no way jose, rule needs to be 0,1,2,3,4")

    return txt


# Problem types
def typeI(stim):
    return stim % 2
def typeII(stim):
    labels = [0,1,1,0,0,1,1,0]
    return(labels[stim])
def typeIII(stim):
    labels = [1,0,1,0,1,1,0,0]
    return(labels[stim])
def typeIV(stim):
    labels = [1,1,1,0,1,0,0,0]
    return(labels[stim])
def typeV(stim):
    labels = [1,0,1,0,1,0,0,1]
    return(labels[stim])
def typeVI(stim):
    labels = [1,0,0,1,0,1,1,0]
    return(labels[stim])

catfuns = [typeI, typeII, typeIII, typeIV, typeV, typeVI]


# SHJ counterbalancing tools as written in javascript by crump/mcdonnell/gureckis and translated to python by me pam

# // We want to be able to alias the order of stimuli to a single number which
# // can be stored and which can easily replicate a given stimulus order.
# /* Extra details: arr is 3 digits which together would be the binary version of decimal number 0 thru 7.
#    E.g., arr = [0,1,0] is binary for decimal 2.
#    Ordernum is either 0 or 1. << WRONG i think it's between 0 and 5 inclusive
# */
def changeorder(arr, ordernum):
    thisorder = ordernum
    shufflelocations = []
    
    for i in range(len(arr)):
        shufflelocations.append(i); # Now shufflelocations = [0,1,2]
    
    for i in reversed(range(len(arr))): # i=2,1,0
        loci = shufflelocations[i]
        locj = shufflelocations[int(thisorder%(i+1))] # [0, 1] modulo [3, 2, 1]
        # Note that 0 mod anything is 0
        # whereas 1%3=1, 1%2=1, 1%1=0
        thisorder = np.floor(thisorder/(i+1)) # for i=2,1,0 then thisorder=0,0,1 for ordernum=1, and 0,0,0 otherwise
        tempi = arr[loci]
        tempj = arr[locj]
        arr[loci] = tempj
        arr[locj] = tempi
    
    return arr

# // Stimulus counterbalancer
def getstim(theorystim,condition):
    assert theorystim < 8, "Stim >=8 ("+theorystim+")"
    assert theorystim >= 0, "Stim less than 0 ("+theorystim+")"
    flippedstim = theorystim^condition['dimvals'] # Here the stim identities are flipped
#         /* Detail on the above line: ^ is bitwise XOR operator. 
#         / dimvals is also between 0 and 7, and theorystim is between 0 and 7 inclusive. 
#         / When dimvals is 0, flippedstim will be the same decimal number as theorystim.
#         / When dimvals is 1, flippedstim will swap 0 with 1 and vice versa, 
#         / 2 with 3 and vice versa,
#         / 4 with 5 et,
#         / 6 with 7. 
#         / So we have these groups [0,1] [2,3] [4,5] [6,7] where flippedstim will be the 
#         / partner of the original value of theorystim IF dimvals is 1. 
#         */

    bits = []
    for i in range(3):
        if flippedstim&(2**i): # if this is nonzero
            bits.append(1)
        else:
            bits.append(0)
#         /* Ampersand & is bitwise AND operator.
#             note: flippedstim is between 0 and 7 inclusive
#             note: Math.pow(2,i) == 2**i and 0<i<3 so the expression has possible decimal values 1,2,4 (in binary, 1, 10, 100)
#             x & 1 returns 1 for odd numbers x=[1,3,5,7] else 0
#             x & 2 returns 2 for x=[2,3,6,7] else 0
#             x & 4 returns 4 for x=[4,5,6,7] else 0
#             The ? 1:0 portion just means change any "1,2,4" result to simply 1, else 0 
#            SO, bits contains the category (0 or 1) of the stimulus under these three rules (right?)
#             e.g., for 
#             x=0 bits=[0,0,0]; x=1 bits=[1,0,0]
#             x=2 bits=[0,1,0]; x=3 bits=[1,1,0]
#             x=4 bits=[0,0,1]; x=5 bits=[1,0,1]
#             x=6 bits=[0,1,1]; x=7 bits=[1,1,1]
#         */

    newbits = changeorder(bits, condition['dimorder']) # in javascript, no new var here bc function works in-place

    multiples = [1, 2, 4]
    ret = 0
    for i in range(3):
        ret += multiples[i] * newbits[i]; # Here we convert from binary [bits] to decimal
    
    return ret

if __name__ == '__main__':
    for rule in [0,1,2,3,4]:
        for dimo in [0,1,2,3,4,5]:
            for dimv in [0,1,2,3,4,5,6,7]: 
                print(ruleToText(rule,dimo,dimv))