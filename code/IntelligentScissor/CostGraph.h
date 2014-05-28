
#ifndef COSTGRAPH_H_
#define COSTGRAPH_H_

#include "../Header.h"
#include "CostNode.h"
#include <vector>
#include "CostNode.h"
#include "fibheap.h"

class CostGraph {
public:
    CostGraph(QImage image);
	virtual ~CostGraph();

	void computeCostLink();
	void computeMaxD();
    void initState();
    void initPrev();
    void initDistance();
	void computeCost();

    void liveWireDP(int seedr, int seedc);

    // store a path in _path[]
    void computePath(int current_r, int current_c);
    std::vector<CostNode*> constructNeibors(CostNode* node);

    void constructTreeImage();


    int _height, _width;
	float _maxD;
    std::vector<std::vector<CostNode*> > _graphAll;
    FibHeap *pHeap;
    QImage *_image, *_treeImage;
    std::vector<CostNode*> _path;
    int _seedr, _seedc;
    int _preSeedr, _preSeedc;

};

#endif /* COSTGRAPH_H_ */
