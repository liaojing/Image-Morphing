#include <iostream>
#include <vector>
#include <QtGui>
#include "CostNode.h"

CostNode::CostNode(int i, int j) : FibHeapNode(), _totalCost(0.0), _state(INITIAL){
	// TODO Auto-generated constructor stub
    _row = i;
    _col = j;
    _prevNode = NULL;
    _distance = 0;
	_links = std::vector<float>(8);
	for (int i = 0; i < 8; i++)
		_links[i] = 0;
}

CostNode::~CostNode() {
	// TODO Auto-generated destructor stub
}

CostNode::CostNode(QImage& image, int i, int j): FibHeapNode(), _totalCost(0.0), _state(INITIAL)
{
    _height = image.height();
    _width = image.width();
    _prevNode = NULL;
    _row = i;
    _col = j;
	std::vector<std::vector<QRgb> > mat(3, std::vector<QRgb>(3));
	//std::vector<float> _links(8);
	_links = std::vector<float>(8);
	for (int k = 0; k < 0; k++)
		_links[k] = 0;

	computeCostLink(image, i, j);
	
}
void CostNode::computeCostLink(QImage& image, int r, int c)
{
	int height = image.height();
	int width = image.width();
	std::vector<std::vector<float> > adj(8, std::vector<float>(3));
	//std::vector<QColor> adj(8);

	if (r == 0 || c == 0 || r == height-1 || c == width-1)
		for (int k = 0; k < 8; k++)
			for (int d = 0; d < 3; d++)
				adj[k][d] = 0;
	else
	{
		QRgb rgb=image.pixel(c+1, r);
		adj[0][0] = qRed(rgb);
		adj[0][1] = qGreen(rgb);
		adj[0][2] = qBlue(rgb);

		rgb=image.pixel(c+1, r-1);
		adj[1][0] = qRed(rgb);
		adj[1][1] = qGreen(rgb);
		adj[1][2] = qBlue(rgb);

		rgb=image.pixel(c, r-1);
		adj[2][0] = qRed(rgb);
		adj[2][1] = qGreen(rgb);
		adj[2][2] = qBlue(rgb);

		rgb=image.pixel(c-1, r-1);
		adj[3][0] = qRed(rgb);
		adj[3][1] = qGreen(rgb);
		adj[3][2] = qBlue(rgb);

		rgb=image.pixel(c-1, r);
		adj[4][0] = qRed(rgb);
		adj[4][1] = qGreen(rgb);
		adj[4][2] = qBlue(rgb);

		rgb=image.pixel(c-1, r+1);
		adj[5][0] = qRed(rgb);
		adj[5][1] = qGreen(rgb);
		adj[5][2] = qBlue(rgb);

		rgb=image.pixel(c, r+1);
		adj[6][0] = qRed(rgb);
		adj[6][1] = qGreen(rgb);
		adj[6][2] = qBlue(rgb);

		rgb=image.pixel(c+1, r+1);
		adj[7][0] = qRed(rgb);
		adj[7][1] = qGreen(rgb);
		adj[7][2] = qBlue(rgb);
	}

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[2][d] + adj[1][d] - adj[6][d] - adj[7][d]) / 4.0f;
		_links[0] += temp * temp;
	}
	_links[0] = sqrt(_links[0]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[2][d] + adj[3][d] - adj[6][d] - adj[5][d]) / 4.0f;
		_links[4] += temp * temp;
	}
	_links[4] = sqrt(_links[4]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[3][d] + adj[4][d] - adj[0][d] - adj[1][d]) / 4.0f;
		_links[2] += temp * temp;
	}
	_links[2] = sqrt(_links[2]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[4][d] + adj[5][d] - adj[0][d] - adj[7][d]) / 4.0f;
		_links[6] += temp * temp;
	}
	_links[6] = sqrt(_links[6]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[0][d] - adj[2][d]) / 1.41421356f;
		_links[1] += temp * temp;
	}
	_links[1] = sqrt(_links[1]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[4][d] - adj[2][d]) /1.41421356f;
		_links[3] += temp * temp;
	}
	_links[3] = sqrt(_links[3]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[4][d] - adj[6][d]) / 1.41421356f;
		_links[5] += temp * temp;
	}
	_links[5] = sqrt(_links[5]);

	for (int d = 0; d < 3; d++)
	{
		float temp = (adj[0][d] - adj[6][d]) / 1.41421356f;
		_links[7] += temp * temp;
	}
	_links[7] = sqrt(_links[7]);

}

void CostNode::Print()
{
    FibHeapNode::Print();
    qDebug() << _row << "  " << _col << " cost: " <<_totalCost;
}
void CostNode::operator =(float NewKeyVal)
{
    qDebug() << " = sigh is not available";
    CostNode temp(_row, _col);
    temp._totalCost = this->_totalCost = NewKeyVal;
    FHN_Assign(temp);
}
void CostNode::operator =(FibHeapNode& RHS)
{
     FHN_Assign(RHS);
     _totalCost = ((CostNode&) RHS)._totalCost;
}
int CostNode::operator ==(FibHeapNode& RHS)
{
    if (FHN_Cmp(RHS)) return 0;
    return _totalCost == ((CostNode&) RHS)._totalCost ? 1:0;
}
int CostNode::operator <(FibHeapNode& RHS)
{
    int x;
    if ((x=FHN_Cmp(RHS)) != 0)
        return x < 0 ? 1:0;
    return _totalCost < ((CostNode&) RHS)._totalCost ? 1:0;
}
CostNode* CostNode::copy()
{
    CostNode* newNode = new CostNode(_row, _col);
    newNode->_col = this->_col;
    newNode->_row = this->_row;
    newNode->_state = this->_state;
    newNode->_totalCost = this->_totalCost;
    newNode->_prevNode = this->_prevNode;
    newNode->_links = this->_links;
    //newNode->_neighbors = this->_neighbors;
    return newNode;
}

CostNode::NeighborType CostNode::getNeighborType(CostNode* n)
{
    int nr = n->_row;
    int nc = n->_col;

    if (nr == _row - 1 && nc == _col -1)
        return CostNode::LU;
    else if (nr == _row - 1 && nc == _col)
        return CostNode::U;
    else if (nr == _row - 1 && nc == _col + 1)
        return CostNode::RU;
    else if (nr == _row && nc == _col - 1)
        return CostNode::L;
    else if (nr == _row && nc == _col + 1)
        return CostNode::R;
    else if (nr == _row + 1 && nc == _col - 1)
        return CostNode::LD;
    else if (nr == _row + 1 && nc == _col)
        return CostNode::D;
    else //if (nr == _row + 1 && nc == _col + 1)
        return CostNode::RD;
}
