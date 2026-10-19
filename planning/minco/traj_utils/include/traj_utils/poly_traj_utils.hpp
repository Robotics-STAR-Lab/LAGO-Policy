#pragma once

#include <iostream>
#include <cmath>
#include <cfloat>
#include <vector>

#include <Eigen/Eigen>

#include "traj_utils/root_finder.hpp"

namespace poly_traj
{
    // Joint-space trajectory for a robotic arm
    // Degree-7 polynomial used to minimize snap
    template <int degree, int traj_dim>
    class JointPiece
    {
    public:
        typedef Eigen::Matrix<double, traj_dim, degree + 1> CoefficientMat;
        typedef Eigen::Matrix<double, traj_dim, degree> VelCoefficientMat;
        typedef Eigen::Matrix<double, traj_dim, degree - 1> AccCoefficientMat;

    private:
        double duration;
        CoefficientMat coeffMat;

    public:
        JointPiece() = default;

        JointPiece(double dur, const CoefficientMat &cMat)
            : duration(dur), coeffMat(cMat) {}

        inline int getDim() const { return traj_dim; }
        inline int getDegree() const { return degree; }
        inline double getDuration() const { return duration; }
        inline const CoefficientMat &getCoeffMat() const { return coeffMat; }

        inline Eigen::Matrix<double, traj_dim, 1> getPos(const double &t) const
        {
            Eigen::Matrix<double, traj_dim, 1> pos;
            pos.setZero();
            double tn = 1.0;
            for (int i = degree; i >= 0; i--)
            {
                pos += tn * coeffMat.col(i);
                tn *= t;
            }
            return pos;
        }

        inline Eigen::Matrix<double, traj_dim, 1> getVel(const double &t) const
        {
            Eigen::Matrix<double, traj_dim, 1> vel;
            vel.setZero();
            double tn = 1.0;
            int n = 1;
            for (int i = degree - 1; i >= 0; i--)
            {
                vel += n * tn * coeffMat.col(i);
                tn *= t;
                n++;
            }
            return vel;
        }

        inline Eigen::Matrix<double, traj_dim, 1> getAcc(const double &t) const
        {
            Eigen::Matrix<double, traj_dim, 1> acc;
            acc.setZero();
            double tn = 1.0;
            int m = 1;
            int n = 2;
            for (int i = degree - 2; i >= 0; i--)
            {
                acc += m * n * tn * coeffMat.col(i);
                tn *= t;
                m++;
                n++;
            }
            return acc;
        }

        inline Eigen::Matrix<double, traj_dim, 1> getJer(const double &t) const
        {
            Eigen::Matrix<double, traj_dim, 1> jer;
            jer.setZero();
            double tn = 1.0;
            int l = 1, m = 2, n = 3;
            for (int i = degree - 3; i >= 0; i--)
            {
                jer += l * m * n * tn * coeffMat.col(i);
                tn *= t;
                l++; m++; n++;
            }
            return jer;
        }

        inline Eigen::Matrix<double, traj_dim, 1> getSnap(const double &t) const
        {
            Eigen::Matrix<double, traj_dim, 1> snap;
            snap.setZero();
            double tn = 1.0;
            int k = 1, l = 2, m = 3, n = 4;
            for (int i = degree - 4; i >= 0; i--)
            {
                snap += k * l * m * n * tn * coeffMat.col(i);
                tn *= t;
                k++; l++; m++; n++;
            }
            return snap;
        }

        inline double getMaxVelRate() const
        {
            VelCoefficientMat nVelCoeffMat = normalizeVelCoeffMat();
            Eigen::VectorXd coeff = Eigen::VectorXd::Zero(2 * degree - 1);
            for (int d = 0; d < traj_dim; d++)
            {
                coeff += RootFinder::polySqr(nVelCoeffMat.row(d).transpose());
            }
            int N = coeff.size();
            int n = N - 1;
            for (int i = 0; i < N; i++)
            {
                coeff(i) *= n;
                n--;
            }
            if (coeff.head(N - 1).squaredNorm() < DBL_EPSILON)
            {
                return getVel(0.0).norm();
            }
            else
            {
                double l = -0.0625;
                double r = 1.0625;
                while (fabs(RootFinder::polyVal(coeff.head(N - 1), l)) < DBL_EPSILON)
                    l = 0.5 * l;
                while (fabs(RootFinder::polyVal(coeff.head(N - 1), r)) < DBL_EPSILON)
                    r = 0.5 * (r + 1.0);
                std::set<double> candidates = RootFinder::solvePolynomial(coeff.head(N - 1), l, r,
                                                                          FLT_EPSILON / duration);
                candidates.insert(0.0);
                candidates.insert(1.0);
                double maxVelRateSqr = -INFINITY;
                for (auto it = candidates.begin(); it != candidates.end(); it++)
                {
                    if (0.0 <= *it && 1.0 >= *it)
                    {
                        double tempNormSqr = getVel((*it) * duration).squaredNorm();
                        maxVelRateSqr = std::max(maxVelRateSqr, tempNormSqr);
                    }
                }
                return sqrt(maxVelRateSqr);
            }
        }

    private:
        inline VelCoefficientMat normalizeVelCoeffMat() const
        {
            VelCoefficientMat nVelCoeffMat;
            int n = 1;
            double t = duration;
            for (int i = degree - 1; i >= 0; i--)
            {
                nVelCoeffMat.col(i) = n * coeffMat.col(i) * t;
                t *= duration;
                n++;
            }
            return nVelCoeffMat;
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    // 关节空间轨迹类
    template <int degree, int traj_dim>
    class JointTrajectory
    {
    private:
        typedef std::vector<JointPiece<degree, traj_dim>> Pieces;
        Pieces pieces;

    public:
        JointTrajectory() = default;

        inline int getPieceNum() const { return pieces.size(); }
        inline int getDim() const { return traj_dim; }

        inline Eigen::VectorXd getDurations() const
        {
            int N = getPieceNum();
            Eigen::VectorXd durations(N);
            for (int i = 0; i < N; i++)
                durations(i) = pieces[i].getDuration();
            return durations;
        }

        inline double getTotalDuration() const
        {
            double totalDuration = 0.0;
            for (const auto &piece : pieces)
                totalDuration += piece.getDuration();
            return totalDuration;
        }

        inline void clear() { pieces.clear(); }
        inline void reserve(int n) { pieces.reserve(n); }

        inline void emplace_back(double dur, const typename JointPiece<degree, traj_dim>::CoefficientMat &cMat)
        {
            pieces.emplace_back(dur, cMat);
        }

        inline int locatePieceIdx(double &t) const
        {
            int N = getPieceNum();
            int idx;
            double dur;
            for (idx = 0; idx < N && t > (dur = pieces[idx].getDuration()); idx++)
                t -= dur;
            if (idx == N)
            {
                idx--;
                t += pieces[idx].getDuration();
            }
            return idx;
        }

        inline Eigen::Matrix<double, traj_dim, 1> getPos(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getPos(t);
        }

        inline Eigen::Matrix<double, traj_dim, 1> getVel(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getVel(t);
        }

        inline Eigen::Matrix<double, traj_dim, 1> getAcc(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getAcc(t);
        }

        inline Eigen::Matrix<double, traj_dim, 1> getJer(double t) const
        {
            int pieceIdx = locatePieceIdx(t);
            return pieces[pieceIdx].getJer(t);
        }

        inline const JointPiece<degree, traj_dim> &operator[](int i) const { return pieces[i]; }
        inline JointPiece<degree, traj_dim> &operator[](int i) { return pieces[i]; }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    // 带状矩阵求解器 (用于MINCO)
    class BandedSystem
    {
    public:
        inline void create(const int &n, const int &p, const int &q)
        {
            destroy();
            N = n;
            lowerBw = p;
            upperBw = q;
            int actualSize = N * (lowerBw + upperBw + 1);
            ptrData = new double[actualSize];
            std::fill_n(ptrData, actualSize, 0.0);
        }

        inline void destroy()
        {
            if (ptrData != nullptr)
            {
                delete[] ptrData;
                ptrData = nullptr;
            }
        }

        inline void reset()
        {
            std::fill_n(ptrData, N * (lowerBw + upperBw + 1), 0.0);
        }

        inline const double &operator()(const int &i, const int &j) const
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline double &operator()(const int &i, const int &j)
        {
            return ptrData[(i - j + upperBw) * N + j];
        }

        inline void factorizeLU()
        {
            int iM, jM;
            double cVl;
            for (int k = 0; k <= N - 2; k++)
            {
                iM = std::min(k + lowerBw, N - 1);
                cVl = operator()(k, k);
                for (int i = k + 1; i <= iM; i++)
                {
                    if (operator()(i, k) != 0.0)
                        operator()(i, k) /= cVl;
                }
                jM = std::min(k + upperBw, N - 1);
                for (int j = k + 1; j <= jM; j++)
                {
                    cVl = operator()(k, j);
                    if (cVl != 0.0)
                    {
                        for (int i = k + 1; i <= iM; i++)
                        {
                            if (operator()(i, k) != 0.0)
                                operator()(i, j) -= operator()(i, k) * cVl;
                        }
                    }
                }
            }
        }

        inline void solve(Eigen::MatrixXd &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; j++)
            {
                iM = std::min(j + lowerBw, N - 1);
                for (int i = j + 1; i <= iM; i++)
                {
                    if (operator()(i, j) != 0.0)
                        b.row(i) -= operator()(i, j) * b.row(j);
                }
            }
            for (int j = N - 1; j >= 0; j--)
            {
                b.row(j) /= operator()(j, j);
                iM = std::max(0, j - upperBw);
                for (int i = iM; i <= j - 1; i++)
                {
                    if (operator()(i, j) != 0.0)
                        b.row(i) -= operator()(i, j) * b.row(j);
                }
            }
        }

        inline void solveAdj(Eigen::MatrixXd &b) const
        {
            int iM;
            for (int j = 0; j <= N - 1; j++)
            {
                b.row(j) /= operator()(j, j);
                iM = std::min(j + upperBw, N - 1);
                for (int i = j + 1; i <= iM; i++)
                {
                    if (operator()(j, i) != 0.0)
                        b.row(i) -= operator()(j, i) * b.row(j);
                }
            }
            for (int j = N - 1; j >= 0; j--)
            {
                iM = std::max(0, j - lowerBw);
                for (int i = iM; i <= j - 1; i++)
                {
                    if (operator()(j, i) != 0.0)
                        b.row(i) -= operator()(j, i) * b.row(j);
                }
            }
        }

    private:
        int N = 0;
        int lowerBw = 0;
        int upperBw = 0;
        double *ptrData = nullptr;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

    // 关节空间MinSnap优化器 (7次多项式)
    // 注意: 函数名getTrajJerkCost保持与原代码一致，实际计算的是Snap代价
    template <int traj_dim_>
    class MinSnapOpt
    {
    public:
        MinSnapOpt() = default;
        ~MinSnapOpt() { A.destroy(); }

    private:
        int N; // pieceNum
        Eigen::MatrixXd headPVAJ, tailPVAJ;
        Eigen::MatrixXd b;
        BandedSystem A;
        Eigen::VectorXd T1, T2, T3, T4, T5, T6, T7;
        Eigen::MatrixXd gdC;

    private:
        template <typename EIGENVEC>
        inline void addGradJbyT(EIGENVEC &gdT) const
        {
            for (int i = 0; i < N; i++)
            {
                gdT(i) += 576.0 * b.row(8 * i + 4).squaredNorm() +
                          5760.0 * b.row(8 * i + 4).dot(b.row(8 * i + 5)) * T1(i) +
                          14400.0 * b.row(8 * i + 5).squaredNorm() * T2(i) +
                          17280.0 * b.row(8 * i + 4).dot(b.row(8 * i + 6)) * T2(i) +
                          86400.0 * b.row(8 * i + 5).dot(b.row(8 * i + 6)) * T3(i) +
                          40320.0 * b.row(8 * i + 4).dot(b.row(8 * i + 7)) * T3(i) +
                          129600.0 * b.row(8 * i + 6).squaredNorm() * T4(i) +
                          201600.0 * b.row(8 * i + 5).dot(b.row(8 * i + 7)) * T4(i) +
                          604800.0 * b.row(8 * i + 6).dot(b.row(8 * i + 7)) * T5(i) +
                          705600.0 * b.row(8 * i + 7).squaredNorm() * T6(i);
            }
        }

        template <typename EIGENMAT>
        inline void addGradJbyC(EIGENMAT &gdC) const
        {
            for (int i = 0; i < N; i++)
            {
                gdC.row(8 * i + 7) += 10080.0 * b.row(8 * i + 4) * T4(i) +
                                      40320.0 * b.row(8 * i + 5) * T5(i) +
                                      100800.0 * b.row(8 * i + 6) * T6(i) +
                                      201600.0 * b.row(8 * i + 7) * T7(i);
                gdC.row(8 * i + 6) += 5760.0 * b.row(8 * i + 4) * T3(i) +
                                      21600.0 * b.row(8 * i + 5) * T4(i) +
                                      51840.0 * b.row(8 * i + 6) * T5(i) +
                                      100800.0 * b.row(8 * i + 7) * T6(i);
                gdC.row(8 * i + 5) += 2880.0 * b.row(8 * i + 4) * T2(i) +
                                      9600.0 * b.row(8 * i + 5) * T3(i) +
                                      21600.0 * b.row(8 * i + 6) * T4(i) +
                                      40320.0 * b.row(8 * i + 7) * T5(i);
                gdC.row(8 * i + 4) += 1152.0 * b.row(8 * i + 4) * T1(i) +
                                      2880.0 * b.row(8 * i + 5) * T2(i) +
                                      5760.0 * b.row(8 * i + 6) * T3(i) +
                                      10080.0 * b.row(8 * i + 7) * T4(i);
            }
        }

        inline void solveAdjGradC(Eigen::MatrixXd &gdC) const
        {
            A.solveAdj(gdC);
        }

        template <typename EIGENVEC>
        inline void addPropCtoT(const Eigen::MatrixXd &adjGdC, EIGENVEC &gdT) const
        {
            Eigen::Matrix<double, 8, traj_dim_> B1;
            Eigen::Matrix<double, 4, traj_dim_> B2;
            for (int i = 0; i < N - 1; i++)
            {
                // negative velocity
                B1.row(3) = -(b.row(i * 8 + 1) +
                              2.0 * T1(i) * b.row(i * 8 + 2) +
                              3.0 * T2(i) * b.row(i * 8 + 3) +
                              4.0 * T3(i) * b.row(i * 8 + 4) +
                              5.0 * T4(i) * b.row(i * 8 + 5) +
                              6.0 * T5(i) * b.row(i * 8 + 6) +
                              7.0 * T6(i) * b.row(i * 8 + 7));
                B1.row(4) = B1.row(3);
                // negative acceleration
                B1.row(5) = -(2.0 * b.row(i * 8 + 2) +
                              6.0 * T1(i) * b.row(i * 8 + 3) +
                              12.0 * T2(i) * b.row(i * 8 + 4) +
                              20.0 * T3(i) * b.row(i * 8 + 5) +
                              30.0 * T4(i) * b.row(i * 8 + 6) +
                              42.0 * T5(i) * b.row(i * 8 + 7));
                // negative jerk
                B1.row(6) = -(6.0 * b.row(i * 8 + 3) +
                              24.0 * T1(i) * b.row(i * 8 + 4) +
                              60.0 * T2(i) * b.row(i * 8 + 5) +
                              120.0 * T3(i) * b.row(i * 8 + 6) +
                              210.0 * T4(i) * b.row(i * 8 + 7));
                // negative snap
                B1.row(7) = -(24.0 * b.row(i * 8 + 4) +
                              120.0 * T1(i) * b.row(i * 8 + 5) +
                              360.0 * T2(i) * b.row(i * 8 + 6) +
                              840.0 * T3(i) * b.row(i * 8 + 7));
                // negative crackle
                B1.row(0) = -(120.0 * b.row(i * 8 + 5) +
                              720.0 * T1(i) * b.row(i * 8 + 6) +
                              2520.0 * T2(i) * b.row(i * 8 + 7));
                // negative d_crackle
                B1.row(1) = -(720.0 * b.row(i * 8 + 6) +
                              5040.0 * T1(i) * b.row(i * 8 + 7));
                // negative dd_crackle
                B1.row(2) = -5040.0 * b.row(i * 8 + 7);

                gdT(i) += B1.cwiseProduct(adjGdC.block(8 * i + 4, 0, 8, traj_dim_)).sum();
            }

            // negative velocity
            B2.row(0) = -(b.row(8 * N - 7) +
                          2.0 * T1(N - 1) * b.row(8 * N - 6) +
                          3.0 * T2(N - 1) * b.row(8 * N - 5) +
                          4.0 * T3(N - 1) * b.row(8 * N - 4) +
                          5.0 * T4(N - 1) * b.row(8 * N - 3) +
                          6.0 * T5(N - 1) * b.row(8 * N - 2) +
                          7.0 * T6(N - 1) * b.row(8 * N - 1));
            // negative acceleration
            B2.row(1) = -(2.0 * b.row(8 * N - 6) +
                          6.0 * T1(N - 1) * b.row(8 * N - 5) +
                          12.0 * T2(N - 1) * b.row(8 * N - 4) +
                          20.0 * T3(N - 1) * b.row(8 * N - 3) +
                          30.0 * T4(N - 1) * b.row(8 * N - 2) +
                          42.0 * T5(N - 1) * b.row(8 * N - 1));
            // negative jerk
            B2.row(2) = -(6.0 * b.row(8 * N - 5) +
                          24.0 * T1(N - 1) * b.row(8 * N - 4) +
                          60.0 * T2(N - 1) * b.row(8 * N - 3) +
                          120.0 * T3(N - 1) * b.row(8 * N - 2) +
                          210.0 * T4(N - 1) * b.row(8 * N - 1));
            // negative snap
            B2.row(3) = -(24.0 * b.row(8 * N - 4) +
                          120.0 * T1(N - 1) * b.row(8 * N - 3) +
                          360.0 * T2(N - 1) * b.row(8 * N - 2) +
                          840.0 * T3(N - 1) * b.row(8 * N - 1));

            gdT(N - 1) += B2.cwiseProduct(adjGdC.block(8 * N - 4, 0, 4, traj_dim_)).sum();
        }

        template <typename EIGENMAT>
        inline void addPropCtoP(const Eigen::MatrixXd &adjGdC, EIGENMAT &gdInP) const
        {
            for (int i = 0; i < N - 1; i++)
            {
                gdInP.col(i) = adjGdC.row(8 * i + 7).transpose();
            }
        }

    public:
        inline void reset(const int &pieceNum)
        {
            N = pieceNum;
            A.create(8 * N, 8, 8);
            b.resize(8 * N, traj_dim_);
            T1.resize(N);
            T2.resize(N);
            T3.resize(N);
            T4.resize(N);
            T5.resize(N);
            T6.resize(N);
            T7.resize(N);
            gdC.resize(8 * N, traj_dim_);
        }

        inline void generate(const Eigen::MatrixXd &inPs,
                             const Eigen::VectorXd &ts,
                             const Eigen::MatrixXd &headState,
                             const Eigen::MatrixXd &tailState)
        {
            headPVAJ = headState;
            tailPVAJ = tailState;

            T1 = ts;
            T2 = T1.cwiseProduct(T1);
            T3 = T2.cwiseProduct(T1);
            T4 = T2.cwiseProduct(T2);
            T5 = T4.cwiseProduct(T1);
            T6 = T4.cwiseProduct(T2);
            T7 = T4.cwiseProduct(T3);

            A.reset();
            b.setZero();

            // 起始点约束: pos, vel, acc, jerk
            A(0, 0) = 1.0;
            A(1, 1) = 1.0;
            A(2, 2) = 2.0;
            A(3, 3) = 6.0;
            b.row(0) = headPVAJ.col(0).transpose();
            b.row(1) = headPVAJ.col(1).transpose();
            b.row(2) = headPVAJ.col(2).transpose();
            b.row(3) = headPVAJ.col(3).transpose();

            // 中间段连续性约束
            for (int i = 0; i < N - 1; i++)
            {
                // snap连续性
                A(8 * i + 4, 8 * i + 4) = 24.0;
                A(8 * i + 4, 8 * i + 5) = 120.0 * T1(i);
                A(8 * i + 4, 8 * i + 6) = 360.0 * T2(i);
                A(8 * i + 4, 8 * i + 7) = 840.0 * T3(i);
                A(8 * i + 4, 8 * i + 12) = -24.0;

                // crackle连续性
                A(8 * i + 5, 8 * i + 5) = 120.0;
                A(8 * i + 5, 8 * i + 6) = 720.0 * T1(i);
                A(8 * i + 5, 8 * i + 7) = 2520.0 * T2(i);
                A(8 * i + 5, 8 * i + 13) = -120.0;

                // 6th derivative连续性
                A(8 * i + 6, 8 * i + 6) = 720.0;
                A(8 * i + 6, 8 * i + 7) = 5040.0 * T1(i);
                A(8 * i + 6, 8 * i + 14) = -720.0;

                // waypoint约束
                A(8 * i + 7, 8 * i) = 1.0;
                A(8 * i + 7, 8 * i + 1) = T1(i);
                A(8 * i + 7, 8 * i + 2) = T2(i);
                A(8 * i + 7, 8 * i + 3) = T3(i);
                A(8 * i + 7, 8 * i + 4) = T4(i);
                A(8 * i + 7, 8 * i + 5) = T5(i);
                A(8 * i + 7, 8 * i + 6) = T6(i);
                A(8 * i + 7, 8 * i + 7) = T7(i);

                // position连续性
                A(8 * i + 8, 8 * i) = 1.0;
                A(8 * i + 8, 8 * i + 1) = T1(i);
                A(8 * i + 8, 8 * i + 2) = T2(i);
                A(8 * i + 8, 8 * i + 3) = T3(i);
                A(8 * i + 8, 8 * i + 4) = T4(i);
                A(8 * i + 8, 8 * i + 5) = T5(i);
                A(8 * i + 8, 8 * i + 6) = T6(i);
                A(8 * i + 8, 8 * i + 7) = T7(i);
                A(8 * i + 8, 8 * i + 8) = -1.0;

                // velocity连续性
                A(8 * i + 9, 8 * i + 1) = 1.0;
                A(8 * i + 9, 8 * i + 2) = 2.0 * T1(i);
                A(8 * i + 9, 8 * i + 3) = 3.0 * T2(i);
                A(8 * i + 9, 8 * i + 4) = 4.0 * T3(i);
                A(8 * i + 9, 8 * i + 5) = 5.0 * T4(i);
                A(8 * i + 9, 8 * i + 6) = 6.0 * T5(i);
                A(8 * i + 9, 8 * i + 7) = 7.0 * T6(i);
                A(8 * i + 9, 8 * i + 9) = -1.0;

                // acceleration连续性
                A(8 * i + 10, 8 * i + 2) = 2.0;
                A(8 * i + 10, 8 * i + 3) = 6.0 * T1(i);
                A(8 * i + 10, 8 * i + 4) = 12.0 * T2(i);
                A(8 * i + 10, 8 * i + 5) = 20.0 * T3(i);
                A(8 * i + 10, 8 * i + 6) = 30.0 * T4(i);
                A(8 * i + 10, 8 * i + 7) = 42.0 * T5(i);
                A(8 * i + 10, 8 * i + 10) = -2.0;

                // jerk连续性
                A(8 * i + 11, 8 * i + 3) = 6.0;
                A(8 * i + 11, 8 * i + 4) = 24.0 * T1(i);
                A(8 * i + 11, 8 * i + 5) = 60.0 * T2(i);
                A(8 * i + 11, 8 * i + 6) = 120.0 * T3(i);
                A(8 * i + 11, 8 * i + 7) = 210.0 * T4(i);
                A(8 * i + 11, 8 * i + 11) = -6.0;

                b.row(8 * i + 7) = inPs.col(i).transpose();
            }

            // 终点约束
            A(8 * N - 4, 8 * N - 8) = 1.0;
            A(8 * N - 4, 8 * N - 7) = T1(N - 1);
            A(8 * N - 4, 8 * N - 6) = T2(N - 1);
            A(8 * N - 4, 8 * N - 5) = T3(N - 1);
            A(8 * N - 4, 8 * N - 4) = T4(N - 1);
            A(8 * N - 4, 8 * N - 3) = T5(N - 1);
            A(8 * N - 4, 8 * N - 2) = T6(N - 1);
            A(8 * N - 4, 8 * N - 1) = T7(N - 1);

            A(8 * N - 3, 8 * N - 7) = 1.0;
            A(8 * N - 3, 8 * N - 6) = 2.0 * T1(N - 1);
            A(8 * N - 3, 8 * N - 5) = 3.0 * T2(N - 1);
            A(8 * N - 3, 8 * N - 4) = 4.0 * T3(N - 1);
            A(8 * N - 3, 8 * N - 3) = 5.0 * T4(N - 1);
            A(8 * N - 3, 8 * N - 2) = 6.0 * T5(N - 1);
            A(8 * N - 3, 8 * N - 1) = 7.0 * T6(N - 1);

            A(8 * N - 2, 8 * N - 6) = 2.0;
            A(8 * N - 2, 8 * N - 5) = 6.0 * T1(N - 1);
            A(8 * N - 2, 8 * N - 4) = 12.0 * T2(N - 1);
            A(8 * N - 2, 8 * N - 3) = 20.0 * T3(N - 1);
            A(8 * N - 2, 8 * N - 2) = 30.0 * T4(N - 1);
            A(8 * N - 2, 8 * N - 1) = 42.0 * T5(N - 1);

            A(8 * N - 1, 8 * N - 5) = 6.0;
            A(8 * N - 1, 8 * N - 4) = 24.0 * T1(N - 1);
            A(8 * N - 1, 8 * N - 3) = 60.0 * T2(N - 1);
            A(8 * N - 1, 8 * N - 2) = 120.0 * T3(N - 1);
            A(8 * N - 1, 8 * N - 1) = 210.0 * T4(N - 1);

            b.row(8 * N - 4) = tailPVAJ.col(0).transpose();
            b.row(8 * N - 3) = tailPVAJ.col(1).transpose();
            b.row(8 * N - 2) = tailPVAJ.col(2).transpose();
            b.row(8 * N - 1) = tailPVAJ.col(3).transpose();

            A.factorizeLU();
            A.solve(b);
        }

        inline const Eigen::MatrixXd &get_b() const { return b; }
        inline const Eigen::VectorXd &get_T1() const { return T1; }
        inline Eigen::MatrixXd &get_gdC() { return gdC; }

        // 保持原函数名getTrajJerkCost，实际计算的是Snap代价
        inline double getTrajJerkCost() const
        {
            double energy = 0.0;
            for (int i = 0; i < N; i++)
            {
                energy += 576.0 * b.row(8 * i + 4).squaredNorm() * T1(i) +
                          2880.0 * b.row(8 * i + 4).dot(b.row(8 * i + 5)) * T2(i) +
                          4800.0 * b.row(8 * i + 5).squaredNorm() * T3(i) +
                          5760.0 * b.row(8 * i + 4).dot(b.row(8 * i + 6)) * T3(i) +
                          21600.0 * b.row(8 * i + 5).dot(b.row(8 * i + 6)) * T4(i) +
                          10080.0 * b.row(8 * i + 4).dot(b.row(8 * i + 7)) * T4(i) +
                          25920.0 * b.row(8 * i + 6).squaredNorm() * T5(i) +
                          40320.0 * b.row(8 * i + 5).dot(b.row(8 * i + 7)) * T5(i) +
                          100800.0 * b.row(8 * i + 6).dot(b.row(8 * i + 7)) * T6(i) +
                          100800.0 * b.row(8 * i + 7).squaredNorm() * T7(i);
            }
            return energy;
        }

        inline JointTrajectory<7, traj_dim_> getTraj() const
        {
            JointTrajectory<7, traj_dim_> traj;
            traj.reserve(N);
            for (int i = 0; i < N; i++)
            {
                Eigen::Matrix<double, traj_dim_, 8> coeffs = b.block(8 * i, 0, 8, traj_dim_).transpose().rowwise().reverse();
                traj.emplace_back(T1(i), coeffs);
            }
            return traj;
        }

        inline Eigen::MatrixXd getInitConstrainPoints(const int K) const
        {
            Eigen::MatrixXd pts(traj_dim_, N * K + 1);
            Eigen::Matrix<double, traj_dim_, 1> pos;
            Eigen::Matrix<double, 8, 1> beta0;
            double s1, s2, s3, s4, s5, s6, s7;
            double step;
            int i_dp = 0;
            // N 是轨迹的段数
            for (int i = 0; i < N; ++i)
            {
                Eigen::MatrixXd c = b.block(i * 8, 0, 8, traj_dim_);
                // T1(i) 是第 i 段轨迹的时间
                step = T1(i) / K;
                s1 = 0.0;

                for (int j = 0; j <= K; ++j)
                {
                    s2 = s1 * s1;
                    s3 = s2 * s1;
                    s4 = s2 * s2;
                    s5 = s4 * s1;
                    s6 = s3 * s3;
                    s7 = s6 * s1;
                    beta0 << 1.0, s1, s2, s3, s4, s5, s6, s7;
                    pos = c.transpose() * beta0;
                    pts.col(i_dp) = pos;

                    s1 += step;
                    if (j != K || (j == K && i == N - 1))
                    {
                        ++i_dp;
                    }
                }
            }
            return pts;
        }

        template <typename EIGENVEC, typename EIGENMAT>
        inline void getGrad2TP(EIGENVEC &gdT, EIGENMAT &gdInPs)
        {
            solveAdjGradC(gdC);
            addPropCtoT(gdC, gdT);
            addPropCtoP(gdC, gdInPs);
        }

        template <typename EIGENVEC>
        inline void initGradCost(EIGENVEC &gdT, double &cost)
        {
            gdT.setZero();
            gdC.setZero();
            cost = getTrajJerkCost();
            addGradJbyT(gdT);
            addGradJbyC(gdC);
        }

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    };

} // namespace poly_traj
