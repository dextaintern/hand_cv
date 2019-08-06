import openpyxl



wbfn = 'saves/phalangedatacollec.xlsx'
wb = openpyxl.load_workbook(wbfn)
ws = wb['calib']

for SRC_IM in ["cj_normal.txt","cj_different2.txt"]:#["andrew_r.txt","chengjie_r.txt","chenmeixi_r.txt","chukai_r.txt","conglifu_r.txt","ruimin_r.txt","shuaishuai_r.txt","shuangshishi_r.txt","sunjiayue_r.txt","tangjuxue_r.txt","ts_r.txt","weiguo_r.txt","zhangyouyou_r.txt","zhangzi_r.txt","zhouran_r.txt","zixiang_r.txt"]:
    f = None
    fingers = []
    f = open('tester/'+SRC_IM,'r')
    for i in range(7):
        f.readline()

    min = f.readline().split()

    for i in range(1):
        f.readline()

    max = f.readline().split()


    for i,imin in enumerate(min):
        if i not in [0]: continue ##thumb, include 2

        imax = max[i]

        fingers += [[float(imin),float(imax)]]


    print("saving to excel, " + SRC_IM)
    for m in fingers:
        ws.append([SRC_IM] + m)
    wb.save(wbfn)
