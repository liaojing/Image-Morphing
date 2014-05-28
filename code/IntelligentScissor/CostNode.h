
#ifndef COSTNODE_H_
#define COSTNODE_H_

#include "../Header.h"
#include "fibheap.h"
#define INITIAL 0
#define EXPANDED 1
#define ACTIVE 2

class CostNode : public FibHeapNode{
public:
    enum NeighborType{LU, U, RU, L, R, LD, D, RD};
    CostNode(int i, int j);
	CostNode(QImage &image, int i, int j);
	void computeCostLink(QImage &image, int r, int c);

    CostNode* copy();
	virtual ~CostNode();

    virtual void operator =(FibHeapNode& RHS);
    virtual int  operator ==(FibHeapNode& RHS);
    virtual int  operator <(FibHeapNode& RHS);

    virtual void operator =(float NewKeyVal);
    virtual void Print();
    float GetKeyValue() { return _totalCost; };
    void SetKeyValue(float inkey) { _totalCost = inkey; };

    NeighborType getNeighborType(CostNode*);
	std::vector<float> _links;
    //std::vector<CostNode*> _neighbors;
	int _col, _row;
    int _state;
    CostNode *_prevNode;
    float _totalCost; // total cost from seed
    int _height, _width;
    int _distance;  //count from seed
    NeighborType _prevNodePos;
};

#endif /* COSTNODE_H_ */
