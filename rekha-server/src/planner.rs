use rekha_core::PlanType;

pub struct QueryPlanner {
    pub alpha: f32,
    pub eval_window: usize,
    plan_history: Vec<(PlanType, f64)>,
    query_count: usize,
}

impl QueryPlanner {
    pub fn new(alpha: f32, eval_window: usize) -> Self {
        Self {
            alpha,
            eval_window,
            plan_history: Vec::new(),
            query_count: 0,
        }
    }

    pub fn select_plan(
        &mut self,
        load_variance: f32,
        dim: usize,
        num_dim_groups: u32,
        num_vector_shards: u64,
    ) -> PlanType {
        self.query_count += 1;

        if !self.query_count.is_multiple_of(self.eval_window) && !self.plan_history.is_empty() {
            return self.plan_history.last().unwrap().0;
        }

        let comp_base = dim as f32 * 0.1;
        let vec_comm = num_vector_shards as f32 * 0.5;
        let dim_comm = num_dim_groups as f32 * 3.0;

        let vec_cost = comp_base * (1.0 + load_variance * self.alpha * 10.0) + vec_comm;
        let dim_cost = comp_base + dim_comm + self.alpha * load_variance * 3.0;
        let hybrid_cost = (vec_cost + dim_cost) / 2.0;

        let plan = if vec_cost <= dim_cost && vec_cost <= hybrid_cost {
            PlanType::VectorBased
        } else if dim_cost <= hybrid_cost {
            PlanType::DimensionBased
        } else {
            PlanType::Hybrid
        };

        let estimated_cost = match plan {
            PlanType::VectorBased => vec_cost,
            PlanType::DimensionBased => dim_cost,
            PlanType::Hybrid => hybrid_cost,
        };

        self.plan_history.push((plan, estimated_cost as f64));
        if self.plan_history.len() > self.eval_window {
            self.plan_history.remove(0);
        }

        plan
    }

    pub fn current_plan(&self) -> PlanType {
        self.plan_history
            .last()
            .map(|(p, _)| *p)
            .unwrap_or(PlanType::VectorBased)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_planner_new() {
        let p = QueryPlanner::new(0.1, 1000);
        assert!((p.alpha - 0.1).abs() < 1e-6);
        assert_eq!(p.eval_window, 1000);
    }

    #[test]
    fn test_planner_select_vector_based() {
        let mut p = QueryPlanner::new(0.1, 1);
        let plan = p.select_plan(0.01, 128, 1, 4);
        assert_eq!(plan, PlanType::VectorBased);
    }

    #[test]
    fn test_planner_select_dim_based() {
        let mut p = QueryPlanner::new(0.5, 1);
        let plan = p.select_plan(0.9, 768, 8, 1);
        assert_eq!(plan, PlanType::DimensionBased);
    }

    #[test]
    fn test_planner_current_plan_default() {
        let p = QueryPlanner::new(0.1, 1000);
        assert_eq!(p.current_plan(), PlanType::VectorBased);
    }

    #[test]
    fn test_planner_eval_window() {
        let mut p = QueryPlanner::new(0.1, 3);
        let plan1 = p.select_plan(0.01, 128, 1, 4);
        let plan2 = p.select_plan(0.01, 128, 1, 4);
        let plan3 = p.select_plan(0.01, 128, 1, 4);
        assert_eq!(plan1, plan2);
        assert_eq!(plan2, plan3);
    }
}
