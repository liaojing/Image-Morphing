#define RECT_SIZE 50 //largest region for intelligent scissor
#include <iostream>
#include <vector>
#include <QtGui>

#include "CostGraph.h"
#include "CostNode.h"
#include "fibheap.h"



CostGraph::CostGraph(QImage image) :_height(image.height()), _width(image.width()){
	// TODO Auto-generated constructor stub	
     _image = new QImage(image);
    _graphAll = std::vector<std::vector<CostNode*> > (_height, std::vector<CostNode*>(_width));

}

void CostGraph::computeCost()
{

	#pragma omp parallel for
	for (int i =_seedr-RECT_SIZE;i<=_seedr+RECT_SIZE;i++)
		for (int j = _seedc-RECT_SIZE; j<=_seedc+RECT_SIZE; j++)
		{
			if(i<0||i>_height-1||j<0||j>_width-1)
				continue;
			if ((i == 0 || j == 0 || i == _height-1 || j == _width-1))
				_graphAll[i][j] =new CostNode(i, j);
			else 
				_graphAll[i][j] =new CostNode(*_image, i, j);			
		}

		computeCostLink();
	
};

CostGraph::~CostGraph() {

	// TODO Auto-generated destructor stub
}

void CostGraph::computeMaxD()
{
	float maxD = 0.0;
	#pragma omp parallel for
	for (int i =_seedr-RECT_SIZE;i<=_seedr+RECT_SIZE;i++)
		for (int j = _seedc-RECT_SIZE; j<=_seedc+RECT_SIZE; j++)
		{
			if(i<0||j<0||i>_height-1||j>_width-1)
				continue;
			for (int k = 0; k < 8; k++)
			{
			     if (maxD < _graphAll[i][j]->_links[k])
                    maxD = _graphAll[i][j]->_links[k];
			}
		}
	this->_maxD = maxD;
}
void CostGraph::computeCostLink()
{
	computeMaxD();
	#pragma omp parallel for
	for (int i =_seedr-RECT_SIZE;i<=_seedr+RECT_SIZE;i++)
		for (int j = _seedc-RECT_SIZE; j<=_seedc+RECT_SIZE; j++)
		{
			if(i<0||j<0||i>_height-1||j>_width-1)
				continue;

			for (int k = 0; k < 8; k++)
			{
                float l = _graphAll[i][j]->_links[k];
				float len;
				if (k % 2 == 0)
					len = 1.0;
				else
					len = sqrt(2.0);

                _graphAll[i][j]->_links[k] = (_maxD - l) * len;

			}
		}
}

void CostGraph::initState()
{
	#pragma omp parallel for
	for (int i =_seedr-RECT_SIZE;i<=_seedr+RECT_SIZE;i++)
		for (int j = _seedc-RECT_SIZE; j<=_seedc+RECT_SIZE; j++)
		{
			if(i<0||j<0||i>_height-1||j>_width-1)
				continue;

            _graphAll[i][j]->_state = INITIAL;
		}
}
void CostGraph::initPrev()
{
	#pragma omp parallel for
	for (int i =_seedr-RECT_SIZE;i<=_seedr+RECT_SIZE;i++)
		for (int j = _seedc-RECT_SIZE; j<=_seedc+RECT_SIZE; j++)
		{
			if(i<0||j<0||i>_height-1||j>_width-1)
				continue;
            _graphAll[i][j]->_prevNode = NULL;
		}
}

std::vector<CostNode*> CostGraph::constructNeibors(CostNode* node)
{
    std::vector<CostNode*> _neighbors(8);
    int r = node->_row;
    int c = node->_col;
    if (c == _width-1)
        _neighbors[0] = NULL;
    else
        _neighbors[0] = _graphAll[r][c+1];
    if (r == 0 || c == _width-1)
        _neighbors[1] = NULL;
    else
        _neighbors[1] = _graphAll[r-1][c+1];
    if (r == 0)
        _neighbors[2] = NULL;
    else
        _neighbors[2] = _graphAll[r-1][c];
    if (r == 0 || c == 0)
        _neighbors[3] = NULL;
    else
        _neighbors[3] = _graphAll[r-1][c-1];
    if (c == 0)
        _neighbors[4] = NULL;
    else
        _neighbors[4] = _graphAll[r][c-1];
    if (r == _height-1 || c == 0)
        _neighbors[5] = NULL;
    else
        _neighbors[5] = _graphAll[r+1][c-1];
    if (r == _height-1)
        _neighbors[6] = NULL;
    else
        _neighbors[6] = _graphAll[r+1][c];
    if (r == _height-1 || c == _width-1)
        _neighbors[7] = NULL;
    else
        _neighbors[7] = _graphAll[r+1][c+1];
    return _neighbors;
}

void CostGraph::liveWireDP(int seedr, int seedc)
{
    int count = 0;

    if (seedr < 0 || seedr > _height-1 || seedc < 0 || seedc > _width-1)
    {
         return;
    }
	seedr=MAX(seedr,0);
	seedr=MIN(seedr,_height-1);
	seedc=MAX(seedc,0);
	seedc=MIN(seedc,_width-1);

    _seedr = seedr;
    _seedc = seedc;

	computeCost();
    
    pHeap = NULL;
	if ((pHeap = new FibHeap) == NULL)
	{
		qDebug() << "memory allocation failed for fheap\n";
		exit(-1);
	}
   
    // initialize
    initState();
    initPrev();
    initDistance();

    // insert seed
     CostNode* seed = _graphAll[seedr][seedc];
    seed->_totalCost = 0.0;
    seed->_prevNode = NULL;
    pHeap->Insert(seed);
    seed->_distance = 0;

    CostNode *q = NULL;
    std::vector<CostNode*> neighbors;
    std::vector<float> neighborCost;
  
    while(pHeap->GetNumNodes()>0)
    {
        q = (CostNode*)pHeap->ExtractMin();
		int r = q->_row;
		int c = q->_col;
		if (r<_seedr-RECT_SIZE||r>_seedr+RECT_SIZE||c<_seedc-RECT_SIZE||c>_seedc+RECT_SIZE)
			continue;

        q->_state = EXPANDED;
        q->_distance = count;
        count++;

       
        neighbors.clear();
        neighborCost.clear();

        // put neighbor info in neighbors
        if (r > 0)
        {
            neighbors.push_back(_graphAll[r-1][c]);
            neighborCost.push_back(q->_links[2]);
            if (c > 0)
            {
                neighbors.push_back(_graphAll[r-1][c-1]);
                neighborCost.push_back(q->_links[3]);
            }
            if (c < _width - 1)
            {
                neighbors.push_back(_graphAll[r-1][c+1]);
                neighborCost.push_back(q->_links[1]);
            }
        }
        if (r < _height - 1)
        {
            neighbors.push_back(_graphAll[r+1][c]);
            neighborCost.push_back(q->_links[6]);
            if (c > 0)
            {
                neighbors.push_back( _graphAll[r+1][c-1]);
                neighborCost.push_back(q->_links[5]);
            }
            if (c < _width - 1)
            {
                neighbors.push_back(_graphAll[r+1][c+1]);
                neighborCost.push_back(q->_links[7]);
            }
        }
        if (c > 0)
        {
            neighbors.push_back(_graphAll[r][c-1]);
            neighborCost.push_back(q->_links[4]);
        }
        if (c < _width - 1)
        {
            neighbors.push_back(_graphAll[r][c+1]);
            neighborCost.push_back(q->_links[0]);
        }

        // for each neibor node r of q
        CostNode* neiborNode;
        for(int i = 0; i < neighbors.size(); i++)
        {
            //qDebug() << "neibor " << i;
            neiborNode = neighbors[i];
            if (neiborNode == NULL)
            {
               continue;
            }

            if (neiborNode->_state == INITIAL)
            {
                
                neiborNode->_prevNode = q;
                neiborNode->_totalCost = q->_totalCost + neighborCost[i];
                neiborNode->_state = ACTIVE;
                pHeap->Insert(neiborNode);
               
            }
            else if (neiborNode->_state == ACTIVE)
            {
                if (q->_totalCost + neighborCost[i] < neiborNode->_totalCost)
                {
                     neiborNode->_prevNode = q;
                    
					 CostNode neiborNodeCp((*_image), neiborNode->_row, neiborNode->_col);
                    neiborNodeCp.SetKeyValue(q->_totalCost + neighborCost[i]);
					neiborNodeCp._state = neiborNode->_state;
                   
                    pHeap->DecreaseKey(neiborNode,neiborNodeCp);
                }
            }
        }

     
    }
   
    delete pHeap;
}

void CostGraph::computePath(int cr, int cc)
{
    // result store in _path
    _path.clear();
    if (cr < 0 || cr > _height-1 || cc < 0 || cc > _width-1)
    {
        qDebug() << "current position out of range";
        return;
    }
	cr=MAX(cr,_seedr-RECT_SIZE);
	cr=MIN(cr,_seedr+RECT_SIZE);
	cc=MAX(cc,_seedc-RECT_SIZE);
	cc=MIN(cc,_seedc+RECT_SIZE);

    CostNode* cNode = _graphAll[cr][cc];
    while(cNode != NULL)
    {
        _path.push_back(cNode);
        cNode = cNode->_prevNode;
    }
}

void CostGraph::initDistance()
{
	#pragma omp parallel for
	for (int i =_seedr-RECT_SIZE;i<=_seedr+RECT_SIZE;i++)
		for (int j = _seedc-RECT_SIZE; j<=_seedc+RECT_SIZE; j++)
		{
			if(i<0||j<0||i>_height-1||j>_width-1)
				continue;
            _graphAll[i][j]->_distance = 65535;
		}
}

