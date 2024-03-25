#include <cstring>
#include <iostream>
#include <fstream>
#include <vector>
#include <cmath>
#include <algorithm>
using namespace std;

const int MAX_TIME = 2000;
const int MAX_TOKEN_SIZE = 10000;
const int MEM_ADAPTER_SIZE = 0;
const int MAX_TOTAL_SIZE = MAX_TOKEN_SIZE + MEM_ADAPTER_SIZE;


class Solver{
public:
    vector <int> dp;
    vector<int> prompt_len;
    vector<int> output_len;
    vector<int> arrive_time;
    vector<int> first_token_latency;
    vector<int> request_pool;
    vector<vector<int>> arrive_reqs;
    vector<vector<int>> end_reqs;
    

    int n;
    int m;
    int serving_num;

    Solver(){
        load_input();
        serving_num=0;

    }
    void load_input(){
        ifstream ifs;
        ifs.open("info.txt", ios::in);
        if (!ifs.is_open()){
            cout<<"error in open file info.txt" << endl;
        }
        else{
            cout<<"success to open file" << endl;
        }
        ifs>>n>>m;

        prompt_len = vector<int>(n, 0);
        output_len = vector<int>(n, 0);
        arrive_time = vector<int>(n, 0);
        first_token_latency = vector<int>(n, 0);
        arrive_reqs = vector<vector<int>>(MAX_TIME, vector<int>(0));
        end_reqs = vector<vector<int>>(MAX_TIME, vector<int>(0));

        for (int i=0;i<n;i++){
            int id, prompt_l, output_l;
            float a_time, f_time;
            ifs >>id>>a_time>>prompt_l>>output_l>>f_time;
            cout<<id<<endl;
            int discretization_time = ceil(a_time*1000/30);
            arrive_reqs[discretization_time].push_back(id);
            prompt_len[id] = prompt_l;
            output_len[id] = output_l;
            arrive_time[id] = a_time;
            first_token_latency[id] = f_time;
        }
    }

    void solve(){
        // dp = vector<int> 
        cout << "solving problem" << endl;
        
        for (int t = 0; t<MAX_TIME; t++){
            // 1. Check if there is a request coming at t
            if (arrive_reqs[t].size() == 0 && end_reqs[t].size() == 0) 
                continue; 
            cout << "requests arrive at " << t << ": ";
            for (auto& req_id : arrive_reqs[t]){
                request_pool.push_back(req_id);
                cout << req_id << ", ";
            }
            cout << endl;
            
            // 2. offload requests and adapters
            serving_num -= end_reqs[t].size();
            cout << "requests arrive at " << t << ": ";
            for (auto& req_id : end_reqs[t]){
                // TODO: free memory space
                cout << req_id << ", ";
            }
            cout << endl;

            // 3. run algorithm to get serve requests

            // vector<int> reqs = dp_algorithm();
            vector<int> reqs = {};    
            
            for (auto& req_id : reqs){
                request_pool.erase(std::remove(request_pool.begin(), request_pool.end(), req_id), request_pool.end());
            }
            serving_num += reqs.size();

            // 4. update the end_time of end_reqs
            for (auto& req_id : reqs){
                end_reqs[t + output_len[req_id]].push_back(req_id);
            }
        }
    }

    vector<int> dp_algorithm(){
        vector<int> reqs;

        return reqs;
    }

private:

};



int main(){
    Solver solver;
    solver.solve();
    return 0;
}