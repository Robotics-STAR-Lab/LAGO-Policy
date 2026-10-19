#ifndef _JOINT_TRAJ_OPTIMIZER_H_
#define _JOINT_TRAJ_OPTIMIZER_H_

#include <Eigen/Eigen>
#include <Eigen/Dense>
#include <memory>
#include <vector>
#include <cmath>
#include <iostream>
#include "optimizer/lbfgs.hpp"
#include "traj_utils/poly_traj_utils.hpp"

namespace joint_traj_opt
{
    // Joint-space trajectory optimizer for a robotic arm
    // N-DOF, degree-7 polynomial minimum-snap implementation
    template <int DOF = 7>
    class JointTrajOptimizer
    {
    public:
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        JointTrajOptimizer() = default;
        ~JointTrajOptimizer() = default;

        // 设置参数
        void setParam(const Eigen::VectorXd &min_pos,
                      const Eigen::VectorXd &max_pos,
                      const Eigen::VectorXd &max_vel,
                      const Eigen::VectorXd &max_acc,
                      double wei_time = 1.0,
                      double wei_feas = 1e3,
                      int cps_per_piece = 5,
                      bool fix_waypoints = false)
        {
            min_joint_pos_ = min_pos;
            max_joint_pos_ = max_pos;
            max_joint_vel_ = max_vel;
            max_joint_acc_ = max_acc;
            wei_time_ = wei_time;
            wei_feas_ = wei_feas;
            cps_num_per_piece_ = cps_per_piece;
            fix_waypoints_ = fix_waypoints;
        }

        // 设置是否固定中间waypoints
        void setFixWaypoints(bool fix) { fix_waypoints_ = fix; }

        // Initialize with default joint parameters (Franka Emika Panda limits)
        void setDefaultFrankaParam()
        {
            min_joint_pos_.resize(DOF);
            max_joint_pos_.resize(DOF);
            max_joint_vel_.resize(DOF);
            max_joint_acc_.resize(DOF);

            // Default joint limits (Franka Emika Panda)
            min_joint_pos_ << -2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973;
            max_joint_pos_ << 2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973;
            max_joint_vel_ << 2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100;
            max_joint_acc_ << 15.0, 7.5, 10.0, 12.5, 15.0, 20.0, 20.0;

            wei_time_ = 1.0;
            wei_feas_ = 1e3;
            cps_num_per_piece_ = 5;
        }

        // 主优化API
        // waypoints: DOF x (N+2) 矩阵，包含起点、终点和N个中间路点
        // initT: 初始时间分配 (N+1维)
        // headState: DOF x 4 矩阵 [pos, vel, acc, jerk]
        // tailState: DOF x 4 矩阵 [pos, vel, acc, jerk]
        bool optimizeTrajectory(const Eigen::MatrixXd &waypoints,
                                const Eigen::VectorXd &initT,
                                const Eigen::MatrixXd &headState,
                                const Eigen::MatrixXd &tailState,
                                Eigen::VectorXd &optT,
                                poly_traj::JointTrajectory<7, DOF> &optTraj)
        {
            if (waypoints.rows() != DOF || waypoints.cols() < 2)
            {
                std::cerr << "Invalid waypoints dimensions!" << std::endl;
                return false;
            }

            piece_num_ = waypoints.cols() - 1;

            // 提取中间路点
            if (piece_num_ > 1)
            {
                innerPts_ = waypoints.middleCols(1, piece_num_ - 1);
            }
            else
            {
                innerPts_.resize(DOF, 0);
            }

            headState_ = headState;
            tailState_ = tailState;

            // 初始化优化器
            minSnapOpt_.reset(piece_num_);

            // 决策变量: 
            // - 如果fix_waypoints_为false: 中间点 + 时间分配
            // - 如果fix_waypoints_为true: 只有时间分配
            int innerPtNum = piece_num_ > 1 ? piece_num_ - 1 : 0;
            
            if (fix_waypoints_)
            {
                // 只优化时间
                variable_num_ = piece_num_;
            }
            else
            {
                // 优化中间点和时间
                variable_num_ = DOF * innerPtNum + piece_num_;
            }

            Eigen::VectorXd x(variable_num_);

            if (!fix_waypoints_)
            {
                // 复制中间点
                if (innerPtNum > 0)
                {
                    for (int i = 0; i < innerPtNum; ++i)
                    {
                        x.segment(i * DOF, DOF) = innerPts_.col(i);
                    }
                }
            }

            // 复制并转换时间分配
            Eigen::VectorXd virtualT(piece_num_);
            realT2VirtualT(initT, virtualT);
            
            int timeOffset = fix_waypoints_ ? 0 : DOF * innerPtNum;
            x.segment(timeOffset, piece_num_) = virtualT;

            // 设置L-BFGS参数
            lbfgs::lbfgs_parameter_t params;
            // params.mem_size = 64;
            params.mem_size = 256;
            // params.g_epsilon = 1e-8;
            params.g_epsilon = 0.0;
            // params.delta = 1e-5;
            params.delta = 1e-3;
            params.past = 3;
            params.max_iterations = 1000;
            // params.min_step = 1e-32;
            params.min_step = 1e-20;

            double finalCost;
            iter_num_ = 0;

            // 运行优化
            int result = lbfgs::lbfgs_optimize(
                x, 
                finalCost,
                &JointTrajOptimizer::costCallback,
                nullptr,
                &JointTrajOptimizer::earlyExitCallback,
                this, 
                params);

            // if (result != lbfgs::LBFGS_CONVERGENCE &&
            //     result != lbfgs::LBFGS_STOP &&
            //     result != lbfgs::LBFGSERR_MAXIMUMLINESEARCH &&
            //     result != lbfgs::LBFGSERR_MAXIMUMITERATION)
            // {
            //     std::cerr << "Optimization failed: " << lbfgs::lbfgs_strerror(result) << std::endl;
            //     return false;
            // }

            if (result != lbfgs::LBFGS_CONVERGENCE && result != lbfgs::LBFGS_STOP && result != lbfgs::LBFGSERR_MAXIMUMLINESEARCH){
                std::cerr << "Optimization failed: " << lbfgs::lbfgs_strerror(result) << std::endl;
            } else if (result == lbfgs::LBFGSERR_MAXIMUMLINESEARCH) {
                std::cout << "Optimization warning: " << lbfgs::lbfgs_strerror(result) << std::endl;
            } else {
                std::cout << "Optimization result is: " << lbfgs::lbfgs_strerror(result) << std::endl;
            }

            if (result == lbfgs::LBFGSERR_INVALID_FUNCVAL) {
                return false;
            }

            // 提取优化结果
            Eigen::VectorXd realT(piece_num_);
            virtualT2RealT(x.segment(timeOffset, piece_num_), realT);
            optT = realT;
            optTraj = minSnapOpt_.getTraj();

            return true;
        }

        // 获取优化轨迹
        poly_traj::JointTrajectory<7, DOF> getOptimizedTrajectory() const
        {
            return minSnapOpt_.getTraj();
        }

        // 获取迭代次数
        int getIterNum() const { return iter_num_; }

        // 获取MinSnapOpt的指针（保持与原代码命名一致）
        inline const poly_traj::MinSnapOpt<DOF> *getMinJerkOptPtr(void) const { return &minSnapOpt_; }

    private:
        // L-BFGS代价函数回调
        static double costCallback(void *instance,
                                   const Eigen::VectorXd &x,
                                   Eigen::VectorXd &g)
        {
            JointTrajOptimizer *opt = reinterpret_cast<JointTrajOptimizer *>(instance);
            return opt->computeCostAndGrad(x, g);
        }

        static int earlyExitCallback(void *instance,
                                     const Eigen::VectorXd &x,
                                     const Eigen::VectorXd &g,
                                     const double fx,
                                     const double step,
                                     const int k,
                                     const int ls)
        {
            return 0; // 不提前退出
        }

        double computeCostAndGrad(const Eigen::VectorXd &x, Eigen::VectorXd &g)
        {
            g.setZero();
            ++iter_num_;

            int innerPtNum = piece_num_ > 1 ? piece_num_ - 1 : 0;
            int timeOffset = fix_waypoints_ ? 0 : DOF * innerPtNum;

            // 解析决策变量
            Eigen::MatrixXd innerPts(DOF, std::max(1, innerPtNum));
            if (innerPtNum > 0)
            {
                if (fix_waypoints_)
                {
                    // 使用固定的中间路点
                    innerPts = innerPts_;
                }
                else
                {
                    // 从决策变量中获取中间路点
                    for (int i = 0; i < innerPtNum; ++i)
                    {
                        innerPts.col(i) = x.segment(i * DOF, DOF);
                    }
                }
            }

            Eigen::VectorXd virtualT = x.segment(timeOffset, piece_num_);
            Eigen::VectorXd realT(piece_num_);
            virtualT2RealT(virtualT, realT);

            // 生成轨迹
            minSnapOpt_.generate(innerPts, realT, headState_, tailState_);

            // 计算平滑性代价 (使用getTrajJerkCost，实际是snap代价)
            double smoothCost = 0;
            Eigen::VectorXd gdT(piece_num_);
            gdT.setZero();
            minSnapOpt_.initGradCost(gdT, smoothCost);

            // 计算可行性代价
            double feasCost = 0;
            computeFeasibilityCost(gdT, feasCost);

            // 总代价
            double totalCost = smoothCost + feasCost;

            // 计算梯度
            Eigen::MatrixXd gdInPs(DOF, std::max(1, innerPtNum));
            gdInPs.setZero();

            if (innerPtNum > 0 && !fix_waypoints_)
            {
                minSnapOpt_.getGrad2TP(gdT, gdInPs);
            }

            // 时间代价及其梯度
            double timeCost = 0;
            Eigen::VectorXd gdVT(piece_num_);
            virtualTGradCost(realT, virtualT, gdT, gdVT, timeCost);

            totalCost += timeCost;

            // 填充梯度
            if (!fix_waypoints_ && innerPtNum > 0)
            {
                for (int i = 0; i < innerPtNum; ++i)
                {
                    g.segment(i * DOF, DOF) = gdInPs.col(i);
                }
            }
            g.segment(timeOffset, piece_num_) = gdVT;

            return totalCost;
        }

        void computeFeasibilityCost(Eigen::VectorXd &gdT, double &cost)
        {
            cost = 0;
            int K = cps_num_per_piece_;

            for (int i = 0; i < piece_num_; ++i)
            {
                Eigen::MatrixXd c = minSnapOpt_.get_b().block(i * 8, 0, 8, DOF);
                double step = minSnapOpt_.get_T1()(i) / K;
                double s1 = 0.0;

                for (int j = 0; j <= K; ++j)
                {
                    double s2 = s1 * s1;
                    double s3 = s2 * s1;
                    double s4 = s2 * s2;
                    double s5 = s4 * s1;
                    double s6 = s3 * s3;
                    double s7 = s4 * s3;

                    Eigen::Matrix<double, 8, 1> beta0, beta1, beta2;
                    beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
                    beta1 << 0.0, 1.0, 2.0 * s1, 3.0 * s2, 4.0 * s3, 5.0 * s4, 6.0 * s5, 7.0 * s6;
                    beta2 << 0.0, 0.0, 2.0, 6.0 * s1, 12.0 * s2, 20.0 * s3, 30.0 * s4, 42.0 * s5;

                    double alpha = (double)j / K;

                    Eigen::Matrix<double, DOF, 1> pos = c.transpose() * beta0;
                    Eigen::Matrix<double, DOF, 1> vel = c.transpose() * beta1;
                    Eigen::Matrix<double, DOF, 1> acc = c.transpose() * beta2;

                    double omg = (j == 0 || j == K) ? 0.5 : 1.0;

                    Eigen::Matrix<double, DOF, 1> gradp, gradv, grada;
                    gradp.setZero();
                    gradv.setZero();
                    grada.setZero();

                    double jointCost = 0;
                    computeJointFeasibilityGrad(pos, vel, acc, gradp, gradv, grada, jointCost);

                    if (jointCost > 0)
                    {
                        Eigen::Matrix<double, 8, DOF> gradViolaPc = beta0 * gradp.transpose();
                        Eigen::Matrix<double, 8, DOF> gradViolaVc = beta1 * gradv.transpose();
                        Eigen::Matrix<double, 8, DOF> gradViolaAc = beta2 * grada.transpose();

                        double gradViolaPt = alpha * gradp.dot(vel);
                        double gradViolaVt = alpha * gradv.dot(acc);

                        minSnapOpt_.get_gdC().block(i * 8, 0, 8, DOF) += omg * step * (gradViolaPc + gradViolaVc + gradViolaAc);
                        gdT(i) += omg * (jointCost / K + step * (gradViolaPt + gradViolaVt));

                        cost += omg * step * jointCost;
                    }

                    s1 += step;
                }
            }
        }

        void computeJointFeasibilityGrad(const Eigen::Matrix<double, DOF, 1> &pos,
                                         const Eigen::Matrix<double, DOF, 1> &vel,
                                         const Eigen::Matrix<double, DOF, 1> &acc,
                                         Eigen::Matrix<double, DOF, 1> &gradp,
                                         Eigen::Matrix<double, DOF, 1> &gradv,
                                         Eigen::Matrix<double, DOF, 1> &grada,
                                         double &cost)
        {
            cost = 0;
            double pen, f, df;

            // 关节位置限制
            for (int i = 0; i < DOF; ++i)
            {
                // 上限
                pen = pos(i) - max_joint_pos_(i);
                if (smoothedL1(pen, 0.01, f, df))
                {
                    cost += wei_feas_ * f;
                    gradp(i) += wei_feas_ * df;
                }

                // 下限
                pen = min_joint_pos_(i) - pos(i);
                if (smoothedL1(pen, 0.01, f, df))
                {
                    cost += wei_feas_ * f;
                    gradp(i) -= wei_feas_ * df;
                }
            }

            // 关节速度限制
            for (int i = 0; i < DOF; ++i)
            {
                pen = vel(i) * vel(i) - max_joint_vel_(i) * max_joint_vel_(i);
                if (smoothedL1(pen, 0.01, f, df))
                {
                    cost += wei_feas_ * f;
                    gradv(i) += wei_feas_ * df * 2.0 * vel(i);
                }
            }

            // 关节加速度限制
            for (int i = 0; i < DOF; ++i)
            {
                pen = acc(i) * acc(i) - max_joint_acc_(i) * max_joint_acc_(i);
                if (smoothedL1(pen, 0.01, f, df))
                {
                    cost += wei_feas_ * f;
                    grada(i) += wei_feas_ * df * 2.0 * acc(i);
                }
            }
        }

        bool smoothedL1(double x, double mu, double &f, double &df)
        {
            if (x < 0.0)
            {
                f = 0;
                df = 0;
                return false;
            }
            else if (x > mu)
            {
                f = x - 0.5 * mu;
                df = 1.0;
                return true;
            }
            else
            {
                double xdmu = x / mu;
                double sqrxdmu = xdmu * xdmu;
                double mumxd2 = mu - 0.5 * x;
                f = mumxd2 * sqrxdmu * xdmu;
                df = sqrxdmu * ((-0.5) * xdmu + 3.0 * mumxd2 / mu);
                return true;
            }
        }

        // 实际时间到虚拟时间的映射
        void realT2VirtualT(const Eigen::VectorXd &RT, Eigen::VectorXd &VT)
        {
            for (int i = 0; i < RT.size(); ++i)
            {
                VT(i) = RT(i) > 1.0 ? (std::sqrt(2.0 * RT(i) - 1.0) - 1.0)
                                    : (1.0 - std::sqrt(2.0 / RT(i) - 1.0));
            }
        }

        // 虚拟时间到实际时间的映射
        void virtualT2RealT(const Eigen::VectorXd &VT, Eigen::VectorXd &RT)
        {
            for (int i = 0; i < VT.size(); ++i)
            {
                RT(i) = VT(i) > 0.0 ? ((0.5 * VT(i) + 1.0) * VT(i) + 1.0)
                                    : 1.0 / ((0.5 * VT(i) - 1.0) * VT(i) + 1.0);
            }
        }

        // 时间代价和梯度
        void virtualTGradCost(const Eigen::VectorXd &RT,
                              const Eigen::VectorXd &VT,
                              const Eigen::VectorXd &gdRT,
                              Eigen::VectorXd &gdVT,
                              double &costT)
        {
            for (int i = 0; i < VT.size(); ++i)
            {
                double gdVT2Rt;
                if (VT(i) > 0)
                {
                    gdVT2Rt = VT(i) + 1.0;
                }
                else
                {
                    double denSqrt = (0.5 * VT(i) - 1.0) * VT(i) + 1.0;
                    gdVT2Rt = (1.0 - VT(i)) / (denSqrt * denSqrt);
                }
                gdVT(i) = (gdRT(i) + wei_time_) * gdVT2Rt;
            }
            costT = RT.sum() * wei_time_;
        }

    private:
        // 优化器
        poly_traj::MinSnapOpt<DOF> minSnapOpt_;

        // 参数
        Eigen::VectorXd min_joint_pos_;
        Eigen::VectorXd max_joint_pos_;
        Eigen::VectorXd max_joint_vel_;
        Eigen::VectorXd max_joint_acc_;
        double wei_time_ = 1.0;
        double wei_feas_ = 1e3;
        int cps_num_per_piece_ = 5;
        bool fix_waypoints_ = false;  // 是否固定中间waypoints

        // 优化状态
        int piece_num_ = 0;
        int variable_num_ = 0;
        int iter_num_ = 0;
        Eigen::MatrixXd innerPts_;
        Eigen::MatrixXd headState_;
        Eigen::MatrixXd tailState_;
    };

} // namespace joint_traj_opt

#endif
