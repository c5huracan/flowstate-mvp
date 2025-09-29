from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Dict, Optional
import json
from datetime import datetime

# ============================================================================
# CORE CLASSES
# ============================================================================

class RequirementMatcher:
    def __init__(self):
        pass
    
    def score_volume_match(self, buyer_volume, seller_capacity):
        if seller_capacity >= buyer_volume:
            return 1.0
        elif seller_capacity >= buyer_volume * 0.5:
            return 0.6
        else:
            return 0.2
    
    def score_latency_match(self, buyer_max_latency, seller_avg_latency):
        if seller_avg_latency <= buyer_max_latency * 0.5:
            return 1.0  # Much better than needed
        elif seller_avg_latency <= buyer_max_latency:
            return 0.8  # Meets requirements
        elif seller_avg_latency <= buyer_max_latency * 1.5:
            return 0.4  # Close but over budget
        else:
            return 0.1  # Significantly too slow
    
    def score_budget_match(self, buyer_budget, seller_pricing, expected_volume):
        # Calculate actual cost based on buyer's expected volume
        if seller_pricing["model"] == "pay_per_use":
            cost_per_1k = seller_pricing["per_1k_calls"]
            # Apply volume discounts if applicable
            for volume_threshold, discount_rate in seller_pricing.get("volume_discounts", {}).items():
                if expected_volume >= volume_threshold:
                    cost_per_1k = discount_rate
            monthly_cost = (expected_volume / 1000) * cost_per_1k
        else:
            monthly_cost = (expected_volume / 1000) * seller_pricing["per_1k_calls"]
        
        # Score based on how it fits buyer's budget
        if monthly_cost <= buyer_budget["max_monthly"] * 0.7:
            return 1.0  # Well within budget
        elif monthly_cost <= buyer_budget["max_monthly"]:
            return 0.8  # Fits budget
        elif monthly_cost <= buyer_budget["max_monthly"] * 1.2:
            return 0.4  # Slightly over budget
        else:
            return 0.1  # Way over budget

class BuyerAgent:
    def __init__(self, requirements):
        self.requirements = requirements
        self.matcher = RequirementMatcher()
        self.evaluation_history = []
        self.negotiation_responses = []
    
    def evaluate_seller(self, seller_agent):
        """Evaluate a seller and return detailed scoring"""
        scores = {}
        
        # Volume matching
        scores['volume'] = self.matcher.score_volume_match(
            self.requirements['expected_volume'],
            seller_agent.capabilities['max_capacity']
        )
        
        # Latency matching  
        scores['latency'] = self.matcher.score_latency_match(
            self.requirements['max_latency'],
            seller_agent.capabilities['avg_latency']
        )
        
        # Budget matching
        scores['budget'] = self.matcher.score_budget_match(
            self.requirements['budget'],
            seller_agent.pricing,
            self.requirements['expected_volume']
        )
        
        # Overall weighted score
        weights = {'volume': 0.3, 'latency': 0.4, 'budget': 0.3}
        overall_score = sum(scores[key] * weights[key] for key in scores)
        
        evaluation = {
            'seller_id': seller_agent.id,
            'scores': scores,
            'overall_score': overall_score,
            'timestamp': datetime.now().isoformat()
        }
        
        self.evaluation_history.append(evaluation)
        return evaluation
    
    def respond_to_counter_proposal(self, seller_id, counter_proposal):
        """Evaluate seller counter-proposals and potentially adjust requirements"""
        response = {"seller_id": seller_id, "interest_level": "low", "reasoning": []}
        
        if 'improvements' in counter_proposal:
            improvements = counter_proposal['improvements']
            
            # Evaluate latency improvements
            if 'latency' in improvements:
                if "50ms" in improvements['latency']:
                    response["interest_level"] = "high"
                    response["reasoning"].append("50ms latency would significantly exceed our requirements")
                    response["follow_up_questions"] = [
                        "What's the additional cost for premium tier?",
                        "What's the SLA guarantee for 50ms response time?",
                        "Is there a middle-tier option between standard and premium?"
                    ]
                elif "premium tier" in improvements['latency'].lower():
                    response["interest_level"] = "medium"
                    response["reasoning"].append("Premium tier could address latency concerns")
            
            # Evaluate budget improvements  
            if 'budget' in improvements:
                if "discount" in improvements['budget'].lower():
                    current_interest = response["interest_level"]
                    if current_interest == "low":
                        response["interest_level"] = "medium"
                    elif current_interest == "medium":
                        response["interest_level"] = "high"
                    
                    response["reasoning"].append("Volume discount makes pricing more competitive")
                    if "follow_up_questions" not in response:
                        response["follow_up_questions"] = []
                    response["follow_up_questions"].extend([
                        "What are the terms of the annual commitment?",
                        "Can we start with 6-month commitment?",
                        "What happens if we don't meet volume commitments?"
                    ])
            
            # Evaluate volume improvements
            if 'volume' in improvements:
                response["reasoning"].append("Scaling capability addresses capacity concerns")
                if response["interest_level"] == "low":
                    response["interest_level"] = "medium"
        
        self.negotiation_responses.append(response)
        return response

class SellerAgent:
    def __init__(self, agent_id, capabilities, pricing):
        self.id = agent_id
        self.capabilities = capabilities
        self.pricing = pricing
        self.proposals = []
    
    def receive_evaluation(self, buyer_evaluation):
        """Receive buyer's evaluation and identify areas for improvement"""
        low_scores = {k: v for k, v in buyer_evaluation['scores'].items() if v < 0.6}
        
        if low_scores:
            return self.generate_counter_proposal(low_scores, buyer_evaluation)
        else:
            return {"status": "accepted", "message": "Requirements well-matched"}
    
    def generate_counter_proposal(self, problem_areas, buyer_evaluation):
        """Generate alternative proposals for low-scoring areas"""
        proposal = {"improvements": {}}
        
        if 'budget' in problem_areas:
            proposal["improvements"]["budget"] = "Can offer 15% volume discount for annual commitment"
        
        if 'latency' in problem_areas:
            proposal["improvements"]["latency"] = "Premium tier offers 50ms guaranteed response time"
            
        if 'volume' in problem_areas:
            proposal["improvements"]["volume"] = "Can scale capacity with 48-hour notice"
        
        self.proposals.append(proposal)
        return proposal

class EnhancedSellerAgent(SellerAgent):
    def respond_to_follow_up(self, buyer_response):
        """Respond to buyer's follow-up questions"""
        responses = {}
        
        if 'follow_up_questions' in buyer_response:
            for question in buyer_response['follow_up_questions']:
                if "additional cost" in question.lower() and "premium" in question.lower():
                    responses[question] = "Premium tier adds $0.005 per 1k calls (total $0.017/1k)"
                elif "sla guarantee" in question.lower():
                    responses[question] = "99.9% uptime with 50ms response time guaranteed or 10% service credit"
                elif "middle-tier" in question.lower():
                    responses[question] = "Yes, we offer 'Enhanced' tier at 100ms guaranteed for $0.003 extra per 1k calls"
                elif "annual commitment" in question.lower():
                    responses[question] = "12-month minimum, paid quarterly, with 30-day termination clause"
                elif "6-month commitment" in question.lower():
                    responses[question] = "6-month available with 10% discount instead of 15%"
                elif "volume commitments" in question.lower():
                    responses[question] = "No penalties for under-usage, discounts apply to actual usage"
        
        return {
            "seller_id": self.id,
            "follow_up_responses": responses,
            "next_steps": "Ready to provide detailed proposal based on your preferred configuration"
        }

class OrchestrationEngine:
    def __init__(self):
        self.discovery_sessions = {}
        self.session_counter = 0
    
    def start_discovery_session(self, buyer_agent, seller_agents):
        """Initialize a collaborative discovery session"""
        session_id = f"session_{self.session_counter}"
        self.session_counter += 1
        
        session = {
            'id': session_id,
            'buyer': buyer_agent,
            'sellers': seller_agents,
            'phase': 'mutual_disclosure',
            'negotiations': [],
            'final_recommendations': []
        }
        
        self.discovery_sessions[session_id] = session
        return self.run_discovery_phase(session_id)
    
    def run_discovery_phase(self, session_id):
        """Execute the collaborative discovery protocol"""
        session = self.discovery_sessions[session_id]
        buyer = session['buyer']
        
        results = []
        
        # Phase 1: Evaluate all sellers
        for seller in session['sellers']:
            evaluation = buyer.evaluate_seller(seller)
            counter_proposal = seller.receive_evaluation(evaluation)
            
            results.append({
                'seller_id': seller.id,
                'initial_evaluation': evaluation,
                'seller_response': counter_proposal
            })
        
        session['negotiations'] = results
        session['phase'] = 'negotiation_complete'
        
        return self.generate_recommendations(session_id)
    
    def generate_recommendations(self, session_id):
        """Generate final recommendations based on collaborative discovery"""
        session = self.discovery_sessions[session_id]
        negotiations = session['negotiations']
        
        # Rank sellers by overall score and improvement potential
        recommendations = []
        for negotiation in negotiations:
            score = negotiation['initial_evaluation']['overall_score']
            has_improvements = 'improvements' in negotiation['seller_response']
            
            recommendations.append({
                'seller_id': negotiation['seller_id'],
                'base_score': score,
                'has_counter_proposal': has_improvements,
                'recommendation_tier': self._calculate_tier(score, has_improvements)
            })
        
        # Sort by recommendation tier and score
        recommendations.sort(key=lambda x: (x['recommendation_tier'], x['base_score']), reverse=True)
        session['final_recommendations'] = recommendations
        
        return recommendations
    
    def _calculate_tier(self, score, has_improvements):
        """Calculate recommendation tier (1=best, 3=worst)"""
        if score >= 0.8:
            return 1
        elif score >= 0.6 or (score >= 0.4 and has_improvements):
            return 2
        else:
            return 3

class CollaborativeDiscoveryPlatform:
    def __init__(self):
        self.sessions = {}
        self.buyers = {}
        self.sellers = {}
        self.session_counter = 0
    
    def register_buyer(self, buyer_id: str, requirements: Dict) -> Dict:
        """Register a new buyer on the platform"""
        buyer = BuyerAgent(requirements)
        self.buyers[buyer_id] = {
            'agent': buyer,
            'registration_date': datetime.now().isoformat(),
            'active_sessions': []
        }
        return {"status": "success", "buyer_id": buyer_id}
    
    def register_seller(self, seller_id: str, capabilities: Dict, pricing: Dict) -> Dict:
        """Register a new seller on the platform"""
        seller = EnhancedSellerAgent(seller_id, capabilities, pricing)
        self.sellers[seller_id] = {
            'agent': seller,
            'registration_date': datetime.now().isoformat(),
            'active_sessions': []
        }
        return {"status": "success", "seller_id": seller_id}
    
    def create_discovery_session(self, buyer_id: str, seller_ids: List[str]) -> Dict:
        """Create a new collaborative discovery session"""
        if buyer_id not in self.buyers:
            return {"status": "error", "message": "Buyer not registered"}
        
        missing_sellers = [sid for sid in seller_ids if sid not in self.sellers]
        if missing_sellers:
            return {"status": "error", "message": f"Sellers not registered: {missing_sellers}"}
        
        session_id = f"session_{self.session_counter}"
        self.session_counter += 1
        
        buyer_agent = self.buyers[buyer_id]['agent']
        seller_agents = [self.sellers[sid]['agent'] for sid in seller_ids]
        
        # Use our existing orchestration engine
        engine = OrchestrationEngine()
        recommendations = engine.start_discovery_session(buyer_agent, seller_agents)
        
        # Store session data
        session_data = {
            'id': session_id,
            'buyer_id': buyer_id,
            'seller_ids': seller_ids,
            'created_at': datetime.now().isoformat(),
            'status': 'completed',
            'recommendations': recommendations,
            'engine': engine  # Store for follow-up negotiations
        }
        
        self.sessions[session_id] = session_data
        
        # Update buyer and seller records
        self.buyers[buyer_id]['active_sessions'].append(session_id)
        for seller_id in seller_ids:
            self.sellers[seller_id]['active_sessions'].append(session_id)
        
        return {
            "status": "success",
            "session_id": session_id,
            "recommendations": recommendations
        }
    
    def get_session_details(self, session_id: str) -> Dict:
        """Retrieve detailed session information"""
        if session_id not in self.sessions:
            return {"status": "error", "message": "Session not found"}
        
        session = self.sessions[session_id]
        engine = session['engine']
        discovery_session = engine.discovery_sessions['session_0']
        
        return {
            "status": "success",
            "session_info": {
                'id': session['id'],
                'buyer_id': session['buyer_id'],
                'seller_ids': session['seller_ids'],
                'created_at': session['created_at'],
                'negotiations': discovery_session['negotiations'],
                'recommendations': session['recommendations']
            }
        }

class PlatformAnalytics:
    def __init__(self, platform: CollaborativeDiscoveryPlatform):
        self.platform = platform
    
    def generate_session_insights(self, session_id: str) -> Dict:
        """Generate actionable insights from a discovery session"""
        session_details = self.platform.get_session_details(session_id)
        if session_details['status'] != 'success':
            return session_details
        
        session_info = session_details['session_info']
        negotiations = session_info['negotiations']
        
        insights = {
            'session_id': session_id,
            'overall_metrics': self._calculate_overall_metrics(negotiations),
            'buyer_insights': self._generate_buyer_insights(negotiations),
            'seller_insights': self._generate_seller_insights(negotiations),
            'market_insights': self._generate_market_insights(negotiations)
        }
        
        return {"status": "success", "insights": insights}
    
    def _calculate_overall_metrics(self, negotiations: List[Dict]) -> Dict:
        """Calculate high-level session metrics"""
        scores = [n['initial_evaluation']['overall_score'] for n in negotiations]
        counter_proposals = len([n for n in negotiations if 'improvements' in n['seller_response']])
        
        return {
            'avg_match_score': sum(scores) / len(scores),
            'best_match_score': max(scores),
            'sellers_with_gaps': counter_proposals,
            'perfect_matches': len([s for s in scores if s >= 0.9])
        }
    
    def _generate_buyer_insights(self, negotiations: List[Dict]) -> Dict:
        """Generate insights for the buyer"""
        # Analyze which requirements are hardest to meet
        requirement_gaps = {'volume': 0, 'latency': 0, 'budget': 0}
        
        for negotiation in negotiations:
            scores = negotiation['initial_evaluation']['scores']
            for req, score in scores.items():
                if score < 0.6:
                    requirement_gaps[req] += 1
        
        # Find most problematic requirement
        most_problematic = max(requirement_gaps.items(), key=lambda x: x[1])
        
        insights = {
            'most_challenging_requirement': most_problematic[0] if most_problematic[1] > 0 else None,
            'requirement_flexibility_suggestions': [],
            'market_position': self._assess_buyer_market_position(negotiations)
        }
        
        # Generate flexibility suggestions
        if requirement_gaps['latency'] > 1:
            insights['requirement_flexibility_suggestions'].append(
                "Consider relaxing latency requirements by 20-30% to access more competitive options"
            )
        if requirement_gaps['budget'] > 1:
            insights['requirement_flexibility_suggestions'].append(
                "Budget constraints are limiting options - consider volume commitments for better pricing"
            )
            
        return insights
    
    def _generate_seller_insights(self, negotiations: List[Dict]) -> Dict:
        """Generate insights for sellers"""
        seller_performance = {}
        
        for negotiation in negotiations:
            seller_id = negotiation['seller_id']
            evaluation = negotiation['initial_evaluation']
            
            seller_performance[seller_id] = {
                'overall_score': evaluation['overall_score'],
                'strengths': [k for k, v in evaluation['scores'].items() if v >= 0.8],
                'weaknesses': [k for k, v in evaluation['scores'].items() if v < 0.6],
                'generated_counter_proposal': 'improvements' in negotiation['seller_response']
            }
        
        return seller_performance
    
    def _generate_market_insights(self, negotiations: List[Dict]) -> Dict:
        """Generate market-level insights"""
        # Analyze competitive landscape
        scores_by_seller = {n['seller_id']: n['initial_evaluation']['overall_score'] 
                           for n in negotiations}
        
        market_leader = max(scores_by_seller.items(), key=lambda x: x[1])
        competitive_gap = market_leader[1] - min(scores_by_seller.values())
        
        return {
            'market_leader': market_leader[0],
            'competitive_gap': round(competitive_gap, 2),
            'market_maturity': 'high' if competitive_gap < 0.2 else 'medium' if competitive_gap < 0.4 else 'low',
            'avg_market_score': sum(scores_by_seller.values()) / len(scores_by_seller)
        }
    
    def _assess_buyer_market_position(self, negotiations: List[Dict]) -> str:
        """Assess how well the buyer's requirements align with market offerings"""
        avg_score = sum(n['initial_evaluation']['overall_score'] for n in negotiations) / len(negotiations)
        
        if avg_score >= 0.8:
            return "strong - requirements well-aligned with market offerings"
        elif avg_score >= 0.6:
            return "moderate - some requirements may need adjustment"
        else:
            return "challenging - significant gaps between requirements and market"

# ============================================================================
# FASTAPI APPLICATION
# ============================================================================

app = FastAPI(
    title="FlowState - Collaborative Discovery Platform", 
    version="1.0.0",
    description="AI-powered collaborative discovery that gets enterprise software sales into flowstate"
)

# Add CORS middleware for web frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend domains
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize platform
platform = CollaborativeDiscoveryPlatform()
analytics = PlatformAnalytics(platform)

# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class BuyerRequirements(BaseModel):
    expected_volume: int
    max_latency: int
    budget: Dict
    use_case: str
    compliance_needs: List[str]

class SellerCapabilities(BaseModel):
    max_capacity: int
    avg_latency: int
    model_quality: str
    compliance: List[str]

class SellerPricing(BaseModel):
    model: str
    per_1k_calls: float
    volume_discounts: Dict[int, float]

class SessionRequest(BaseModel):
    buyer_id: str
    seller_ids: List[str]

# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    return {
        "message": "FlowState - Where Discovery Flows Naturally", 
        "version": "1.0.0",
        "docs": "/docs",
        "status": "active"
    }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": datetime.now().isoformat()}

@app.post("/buyers/register")
async def register_buyer(buyer_id: str, requirements: BuyerRequirements):
    """Register a new buyer on the platform"""
    result = platform.register_buyer(buyer_id, requirements.dict())
    if result["status"] == "success":
        return result
    raise HTTPException(status_code=400, detail=result["message"])

@app.post("/sellers/register") 
async def register_seller(seller_id: str, capabilities: SellerCapabilities, pricing: SellerPricing):
    """Register a new seller on the platform"""
    result = platform.register_seller(seller_id, capabilities.dict(), pricing.dict())
    if result["status"] == "success":
        return result
    raise HTTPException(status_code=400, detail=result["message"])

@app.post("/sessions/create")
async def create_session(request: SessionRequest):
    """Create a new collaborative discovery session"""
    result = platform.create_discovery_session(request.buyer_id, request.seller_ids)
    if result["status"] == "success":
        return result
    raise HTTPException(status_code=400, detail=result["message"])

@app.get("/sessions/{session_id}")
async def get_session(session_id: str):
    """Get session details"""
    result = platform.get_session_details(session_id)
    if result["status"] == "success":
        return result
    raise HTTPException(status_code=404, detail=result["message"])

@app.get("/sessions/{session_id}/analytics")
async def get_session_analytics(session_id: str):
    """Get session analytics and insights"""
    result = analytics.generate_session_insights(session_id)
    if result["status"] == "success":
        return result
    raise HTTPException(status_code=404, detail=result["message"])

@app.get("/platform/stats")
async def get_platform_stats():
    """Get overall platform statistics"""
    return {
        "total_buyers": len(platform.buyers),
        "total_sellers": len(platform.sellers),
        "total_sessions": len(platform.sessions),
        "timestamp": datetime.now().isoformat()
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 5001))
    uvicorn.run(app, host="0.0.0.0", port=port)
