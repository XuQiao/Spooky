results = [('NB1', 20, 1, 2, 0.35271836106007565), ('NB1', 20, 1, 3, 0.34381787761973742), ('NB1', 20, 1, 5, 0.31792859442893728), ('NB1', 20, 1, 7, 0.30839459092709259), ('NB1', 20, 2, 2, 0.35349327971851807), ('NB1', 20, 2, 3, 0.34242170744518569), ('NB1', 20, 2, 5, 0.3170762081622488), ('NB1', 20, 2, 7, 0.30429016330922454), ('NB1', 20, 3, 2, 0.34294954731289123), ('NB1', 20, 3, 3, 0.34058980475411055), ('NB1', 20, 3, 5, 0.31483013310022462), ('NB1', 20, 3, 7, 0.3072090343878065), ('NB2', 20, 1, 2, 0.35809878834760595), ('NB2', 20, 1, 3, 0.3499585262897682), ('NB2', 20, 1, 5, 0.32015597809099655), ('NB2', 20, 1, 7, 0.30864919550444969), ('NB2', 20, 2, 2, 0.35962550763808609), ('NB2', 20, 2, 3, 0.34365311585108366), ('NB2', 20, 2, 5, 0.31460500300254057), ('NB2', 20, 2, 7, 0.30338645277747495), ('NB2', 20, 3, 2, 0.34392158511283355), ('NB2', 20, 3, 3, 0.34865320334297167), ('NB2', 20, 3, 5, 0.31396819712935231), ('NB2', 20, 3, 7, 0.30597416278201683), ('NB1', 30, 1, 2, 0.35454770460196128), ('NB1', 30, 1, 3, 0.34166314397238529), ('NB1', 30, 1, 5, 0.3191013221664023), ('NB1', 30, 1, 7, 0.3085567832288012), ('NB1', 30, 2, 2, 0.354932826426828), ('NB1', 30, 2, 3, 0.34199813790011474), ('NB1', 30, 2, 5, 0.31450988152219028), ('NB1', 30, 2, 7, 0.30534016254151242), ('NB1', 30, 3, 2, 0.34451935776619691), ('NB1', 30, 3, 3, 0.33953900869625403), ('NB1', 30, 3, 5, 0.31582595486223519), ('NB1', 30, 3, 7, 0.3038433330327715), ('NB2', 30, 1, 2, 0.35728277941671494), ('NB2', 30, 1, 3, 0.34710466857441352), ('NB2', 30, 1, 5, 0.32079379506057004), ('NB2', 30, 1, 7, 0.30880787339273164), ('NB2', 30, 2, 2, 0.3608233422673236), ('NB2', 30, 2, 3, 0.34352614295687695), ('NB2', 30, 2, 5, 0.31270145592275422), ('NB2', 30, 2, 7, 0.3051536095646874), ('NB2', 30, 3, 2, 0.34500441430219358), ('NB2', 30, 3, 3, 0.34678308030288341), ('NB2', 30, 3, 5, 0.31387091541253004), ('NB2', 30, 3, 7, 0.30353239629299295), ('NB1', 50, 1, 2, 0.35176746788350466), ('NB1', 50, 1, 3, 0.34043062431365101), ('NB1', 50, 1, 5, 0.31698673616466444), ('NB1', 50, 1, 7, 0.30712392781851194), ('NB1', 50, 2, 2, 0.35376833152313619), ('NB1', 50, 2, 3, 0.34153823214174217), ('NB1', 50, 2, 5, 0.31560380328775861), ('NB1', 50, 2, 7, 0.30394821736297783), ('NB1', 50, 3, 2, 0.34332823954468006), ('NB1', 50, 3, 3, 0.3402982032475827), ('NB1', 50, 3, 5, 0.31501178557529463), ('NB1', 50, 3, 7, 0.30274353434632062), ('NB2', 50, 1, 2, 0.35561598139677464), ('NB2', 50, 1, 3, 0.3440895370125609), ('NB2', 50, 1, 5, 0.31919347480311494), ('NB2', 50, 1, 7, 0.30783368991651916), ('NB2', 50, 2, 2, 0.35824255221840906), ('NB2', 50, 2, 3, 0.34085406473061353), ('NB2', 50, 2, 5, 0.31258776575042785), ('NB2', 50, 2, 7, 0.30212369713211568), ('NB2', 50, 3, 2, 0.34234432917997165), ('NB2', 50, 3, 3, 0.34472713063519922), ('NB2', 50, 3, 5, 0.31295315274000057), ('NB2', 50, 3, 7, 0.3024439538329361),('NB2', 50, 2, 5, 0.31262014100156144), ('NB2', 150, 2, 5, 0.31354026229747622),('NB2', 100, 2, 7, 0.3021663351759406), ('NB2', 200, 2, 7, 0.30380306738295593)]

scores = []
for result in results:
    scores.append(result[4])
max_value = min(scores)
max_index = scores.index(max_value)
print(results[max_index])


