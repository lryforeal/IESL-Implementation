#include <bits/stdc++.h>

using namespace std;

const int PlayerNum=3,M=3;
const int T=1e6;
const double epsilon=0.03;
const double eta=1e-4;

const int NodeMax=5e6;
const int InfoMax=5e4;
double pr_private=1.0;
int op,ed;

struct TreeEdge{
	int a,to,next;
}Te[NodeMax*3];

struct TreeNode{
	bool is_leaf;
	int player;
	int box;
	double util[PlayerNum+1];
	double pc;
}Tn[NodeMax],Tn_info[InfoMax];

double Payoff[PlayerNum+1][M][M][M];
double Util[PlayerNum+1];

int NodeCnt,EdgeCnt,GameTreeRoot;
int ImageTreeRoot[PlayerNum+1];

int Infoset[NodeMax];
bool valid[NodeMax][3];
double V[NodeMax],Q[NodeMax][3];

double dp[NodeMax];
bool vis[NodeMax];
int relink[InfoMax][100];

double score[InfoMax][3];
double policy[InfoMax][3];
double w[InfoMax][3];
double z[InfoMax];

void addnode(int player){
	NodeCnt++;
	Tn[NodeCnt].is_leaf=false;
	Tn[NodeCnt].player=player;
	Tn[NodeCnt].box=0;
}

void addnode_info(int player){
	NodeCnt++;
	Tn_info[NodeCnt].is_leaf=false;
	Tn_info[NodeCnt].player=player;
	Tn_info[NodeCnt].box=0;
}

void addleafnode(double pc){
	NodeCnt++;
	Tn[NodeCnt].is_leaf=true;
	for (int i=1;i<=PlayerNum;i++)
		Tn[NodeCnt].util[i]=Util[i];
	Tn[NodeCnt].pc=pc;
}

void addedge(int s,int t,int action){
	EdgeCnt++;
	Te[EdgeCnt].a=action;
	Te[EdgeCnt].to=t;
	Te[EdgeCnt].next=Tn[s].box;
	Tn[s].box=EdgeCnt;
}

void addedge_info(int s,int t,int action){
	EdgeCnt++;
	Te[EdgeCnt].a=action;
	Te[EdgeCnt].to=t;
	Te[EdgeCnt].next=Tn_info[s].box;
	Tn_info[s].box=EdgeCnt;
}

void init_matrix(){
	for (int n=1;n<=PlayerNum;n++)
		for (int i=0;i<M;i++)
			for (int j=0;j<M;j++)
				for (int k=0;k<M;k++){
					scanf("%lf",&Payoff[n][i][j][k]);
					Payoff[n][i][j][k]*=10;	//corrsponding to the scaling on line 298
				}
}

void buildGameTree(){
	init_matrix();
	addnode(0),GameTreeRoot=NodeCnt;
	addnode(1),addedge(GameTreeRoot,NodeCnt,0);
	int pa1=NodeCnt;
	for (int i=0;i<M;i++){
		addnode(2),addedge(pa1,NodeCnt,i),valid[pa1][i]=true;
		int pa2=NodeCnt;
		for (int j=0;j<M;j++){
			addnode(3),addedge(pa2,NodeCnt,j),valid[pa2][j]=true;
			int pa3=NodeCnt;
			for (int k=0;k<M;k++){
				for (int n=1;n<=PlayerNum;n++) Util[n]=Payoff[n][i][j][k];
				addleafnode(pr_private),addedge(pa3,NodeCnt,k),valid[pa3][k]=true;
			}
		}
	}
}

void traverseGameTree(int curNode,int player,int curParent,int curAction){
	int prevAction=curAction;
	for (int i=Tn[curNode].box;i;i=Te[i].next){
		if (Tn[Te[i].to].is_leaf) continue;
		if (Tn[Te[i].to].player==player){
			int NodeNum=0;
			for (int j=Tn_info[curParent].box;j;j=Te[j].next)
				if (Te[j].a==curAction){
					NodeNum=Te[j].to;
					break;
				}
			if (!NodeNum){
				addnode_info(player);
				addedge_info(curParent,NodeCnt,curAction);
				NodeNum=NodeCnt;
			}
			Infoset[Te[i].to]=NodeNum;
			relink[NodeNum][0]++;
			relink[NodeNum][relink[NodeNum][0]]=Te[i].to;
			traverseGameTree(Te[i].to,player,NodeNum,curAction);
		}
		else traverseGameTree(Te[i].to,player,curParent,curAction);
	}
}

void buildImageTree(){
	op=NodeCnt,NodeCnt=0;
	//printf("%d ",op-1);
	for (int i=1;i<=PlayerNum;i++){
		addnode_info(0);
		ImageTreeRoot[i]=NodeCnt;
		relink[NodeCnt][0]=1;
		relink[NodeCnt][1]=GameTreeRoot;
		traverseGameTree(GameTreeRoot,i,ImageTreeRoot[i],0);
	}
	ed=NodeCnt;
	//printf("%d\n",ed-PlayerNum);
}

void init(int cur){
	double sum=0.0;
	for (int i=0;i<3;i++){
		if (!valid[relink[cur][1]][i]) continue;
		score[cur][i]=0.0;
		sum+=exp(score[cur][i]/epsilon);
	}
	for (int i=0;i<3;i++){
		if (!valid[relink[cur][1]][i]) continue;
		policy[cur][i]=exp(score[cur][i]/epsilon)/sum;
	}
	for (int i=Tn_info[cur].box;i;i=Te[i].next)
		init(Te[i].to);
}

double calc(int player,int cur,double pr,double belief,int iter){
	if (Tn[cur].is_leaf)
		return Tn[cur].util[player];
	double ret=0;
	for (int i=Tn[cur].box;i;i=Te[i].next){
		double t;
		if (cur==GameTreeRoot) t=pr_private;
		else t=policy[Infoset[cur]][Te[i].a];
		double temp;
		if (Tn[cur].player==player){
			temp=calc(player,Te[i].to,pr*t,belief,iter);
			Q[cur][Te[i].a]=temp;
		}
		else temp=calc(player,Te[i].to,pr,belief*t,iter);
		ret+=t*temp;
	}
	if (Tn[cur].player==player){
		for (int i=Tn[cur].box;i;i=Te[i].next)
			w[Infoset[cur]][Te[i].a]+=pr*belief*(Q[cur][Te[i].a]-V[cur]);
		z[Infoset[cur]]+=pr*belief;
	}
	return ret;
}

void update(int cur,int iter){
	double maxi=-1e9;
	for (int i=0;i<3;i++){
		if (!valid[relink[cur][1]][i]) continue;
		if (z[cur]>0.0) w[cur][i]/=z[cur];
		score[cur][i]+=eta*(w[cur][i]-score[cur][i]);
		maxi=max(maxi,score[cur][i]/epsilon);	
	}
	double sum=0.0;
	for (int i=0;i<3;i++){
		if (!valid[relink[cur][1]][i]) continue;
		sum+=exp(score[cur][i]/epsilon-maxi);
	}
	for (int i=0;i<3;i++){
		if (!valid[relink[cur][1]][i]) continue;
		policy[cur][i]=exp(score[cur][i]/epsilon-maxi)/sum;
	}
	for (int i=Tn_info[cur].box;i;i=Te[i].next)
		update(Te[i].to,iter);
}

double re_calc(int player,int cur,double pr,double belief,int iter){
	if (Tn[cur].is_leaf){
		dp[cur]=belief*Tn[cur].util[player];
		vis[cur]=true;
		return Tn[cur].util[player];
	}
	double ret=0;
	for (int i=Tn[cur].box;i;i=Te[i].next){
		double t;
		if (cur==GameTreeRoot) t=pr_private;
		else t=policy[Infoset[cur]][Te[i].a];
		double temp;
		if (Tn[cur].player==player)
			temp=re_calc(player,Te[i].to,pr*t,belief,iter);
		else temp=re_calc(player,Te[i].to,pr,belief*t,iter);
		ret+=t*temp;
	}
	return ret;
}

double query(int cur){
	if (vis[cur]) return dp[cur];
	double ret=0.0;
	for (int i=Tn[cur].box;i;i=Te[i].next)
		ret+=query(Te[i].to);
	return ret;
}

void get_deviation(int cur){
	for (int i=Tn_info[cur].box;i;i=Te[i].next)
		get_deviation(Te[i].to);
	double sum[100][3];
	for (int i=0;i<100;i++)
		for (int j=0;j<3;j++)
			sum[i][j]=0.0;
	double ans=0.0;
	for (int i=1;i<=relink[cur][0];i++){
		int curnode=relink[cur][i],cnt=0;
		for (int j=Tn[curnode].box;j;j=Te[j].next){
			double temp=query(Te[j].to);
			if (Tn_info[cur].player){
				sum[i][Te[j].a]+=temp;
				sum[0][Te[j].a]+=temp;
			}
			else ans+=temp;
		}
	}
	int maxi=2;
	if (Tn_info[cur].player){
		for (int i=0;i<=1;i++)
			if (valid[relink[cur][1]][i]&&sum[0][i]>sum[0][maxi]) maxi=i;
	}
	for (int i=1;i<=relink[cur][0];i++){
		if (Tn_info[cur].player)
			dp[relink[cur][i]]=sum[i][maxi];
		else dp[relink[cur][i]]=ans;
		vis[relink[cur][i]]=true;
	}
}

void solve(){
	for (int i=1;i<=PlayerNum;i++)
		for (int j=Tn_info[ImageTreeRoot[i]].box;j;j=Te[j].next)
			init(Te[j].to);
	for (int iter=1;iter<=T;iter++){
		for (int i=1;i<=ed;i++){
			for (int j=0;j<3;j++)
				w[i][j]=0.0;
			z[i]=0.0;
		}
		for (int i=1;i<=PlayerNum;i++)
			calc(i,GameTreeRoot,1.0,1.0,iter);
		for (int i=1;i<=PlayerNum;i++)
			for (int j=Tn_info[ImageTreeRoot[i]].box;j;j=Te[j].next)
				update(Te[j].to,iter);
		double deviation=0.0;
		if (iter%1000==0){
			for (int i=1;i<=PlayerNum;i++){
				for (int j=1;j<=op;j++){
					dp[j]=0.0;
					vis[j]=false;
				}
				double value=re_calc(i,GameTreeRoot,1.0,1.0,iter);
				get_deviation(ImageTreeRoot[i]);
				deviation+=dp[GameTreeRoot]-value;
			}
			printf("%d %.6lf\n",iter,deviation/10);	//corrsponding to the scaling on line 90
		}
	}
}

int main(){
	freopen("1.txt","r",stdin);
	buildGameTree();
	buildImageTree();
	solve();
	return 0;
}
