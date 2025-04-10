import os
import time
import json
import asyncio
import aiohttp
import random
from typing import Dict, Any, Optional, Tuple, List

class LambdaLabsManager:
    """Manager for academic access to Lambda Labs resources"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or "config/lambda_labs_config.json"
        self.accounts = []
        self.active_sessions = {}
        self.usage_stats = {}
        self.lock = asyncio.Lock()
        self.api_base_url = "https://cloud.lambdalabs.com/api/v1"
        
        # Load configuration
        self._load_config()
    
    def _load_config(self):
        """Load configuration from file"""
        try:
            if os.path.exists(self.config_path):
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    self.accounts = config.get('accounts', [])
                    self.usage_stats = config.get('usage_stats', {})
                print(f"Loaded {len(self.accounts)} Lambda Labs academic accounts")
        except Exception as e:
            print(f"Error loading Lambda Labs configuration: {e}")
            self.accounts = []
            self.usage_stats = {}
    
    def _save_config(self):
        """Save configuration to file"""
        try:
            os.makedirs(os.path.dirname(self.config_path), exist_ok=True)
            with open(self.config_path, 'w') as f:
                json.dump({
                    'accounts': self.accounts,
                    'usage_stats': self.usage_stats
                }, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving Lambda Labs configuration: {e}")
            return False
    
    async def register_academic_account(self, username: str, api_key: str, 
                                     institution: str, email: str) -> bool:
        """Register a new academic account"""
        async with self.lock:
            # Check if account already exists
            for account in self.accounts:
                if account['username'] == username:
                    # Update existing account
                    account.update({
                        'api_key': api_key,
                        'institution': institution,
                        'email': email,
                        'updated_at': time.time()
                    })
                    self._save_config()
                    return True
            
            # Create new account
            self.accounts.append({
                'username': username,
                'api_key': api_key,
                'institution': institution,
                'email': email,
                'created_at': time.time(),
                'updated_at': time.time(),
                'status': 'active'
            })
            
            # Initialize usage stats
            self.usage_stats[username] = {
                'total_usage_hours': 0,
                'monthly_usage_hours': 0,
                'last_reset': time.time(),
                'instances_launched': 0,
                'successful_sessions': 0,
                'failed_sessions': 0
            }
            
            self._save_config()
            return True
    
    async def get_available_account(self, requirements: Optional[Dict[str, Any]] = None) -> Optional[Dict[str, Any]]:
        """Get an available academic account based on requirements"""
        async with self.lock:
            # Reset monthly usage if needed
            current_month = time.strftime('%Y-%m', time.localtime())
            for username, stats in self.usage_stats.items():
                last_reset_month = time.strftime('%Y-%m', time.localtime(stats.get('last_reset', 0)))
                if last_reset_month != current_month:
                    stats['monthly_usage_hours'] = 0
                    stats['last_reset'] = time.time()
            
            # Filter active accounts
            active_accounts = [a for a in self.accounts if a.get('status') == 'active']
            
            if not active_accounts:
                return None
            
            # Sort by monthly usage (ascending)
            sorted_accounts = sorted(active_accounts, 
                                    key=lambda a: self.usage_stats.get(a['username'], {}).get('monthly_usage_hours', 0))
            
            # Return account with lowest usage
            return sorted_accounts[0] if sorted_accounts else None
    
    async def create_instance(self, account: Dict[str, Any], 
                           instance_type: str = "gpu_1x_a10", 
                           region: str = "us-east-1") -> Tuple[bool, str, Dict[str, Any]]:
        """Create a new Lambda Labs instance"""
        # In a real implementation, this would use the Lambda Labs API
        # This is a simplified simulation
        print(f"Creating Lambda Labs instance for account {account['username']}")
        
        # Simulate API call
        instance_id = f"instance-{random.randint(10000, 99999)}"
        instance_ip = f"34.{random.randint(1, 255)}.{random.randint(1, 255)}.{random.randint(1, 255)}"
        
        # Record usage
        async with self.lock:
            stats = self.usage_stats.get(account['username'], {
                'total_usage_hours': 0,
                'monthly_usage_hours': 0,
                'last_reset': time.time(),
                'instances_launched': 0,
                'successful_sessions': 0,
                'failed_sessions': 0
            })
            
            stats['instances_launched'] += 1
            self.usage_stats[account['username']] = stats
            self._save_config()
        
        # Create session info
        session_id = f"lambda-{instance_id}"
        session_info = {
            'session_id': session_id,
            'instance_id': instance_id,
            'instance_ip': instance_ip,
            'instance_type': instance_type,
            'region': region,
            'username': account['username'],
            'started_at': time.time(),
            'status': 'active'
        }
        
        # Store in active sessions
        self.active_sessions[session_id] = session_info
        
        return True, session_id, session_info
    
    async def terminate_instance(self, session_id: str) -> bool:
        """Terminate a Lambda Labs instance"""
        if session_id not in self.active_sessions:
            return False
        
        session = self.active_sessions[session_id]
        
        # Calculate usage
        usage_hours = (time.time() - session['started_at']) / 3600
        
        # Update usage stats
        async with self.lock:
            stats = self.usage_stats.get(session['username'], {
                'total_usage_hours': 0,
                'monthly_usage_hours': 0,
                'last_reset': time.time(),
                'instances_launched': 0,
                'successful_sessions': 0,
                'failed_sessions': 0
            })
            
            stats['total_usage_hours'] += usage_hours
            stats['monthly_usage_hours'] += usage_hours
            stats['successful_sessions'] += 1
            
            self.usage_stats[session['username']] = stats
            self._save_config()
        
        # Remove from active sessions
        del self.active_sessions[session_id]
        
        return True
    
    async def get_account_usage(self, username: str) -> Dict[str, Any]:
        """Get usage statistics for an account"""
        async with self.lock:
            stats = self.usage_stats.get(username, {
                'total_usage_hours': 0,
                'monthly_usage_hours': 0,
                'last_reset': time.time(),
                'instances_launched': 0,
                'successful_sessions': 0,
                'failed_sessions': 0
            })
            
            # Add active sessions
            active_sessions = [s for s in self.active_sessions.values() if s['username'] == username]
            
            return {
                'username': username,
                'usage_stats': stats,
                'active_sessions': len(active_sessions),
                'active_session_details': active_sessions
            }
    
    async def get_all_usage(self) -> Dict[str, Any]:
        """Get usage statistics for all accounts"""
        async with self.lock:
            account_usage = {}
            
            for account in self.accounts:
                username = account['username']
                stats = self.usage_stats.get(username, {
                    'total_usage_hours': 0,
                    'monthly_usage_hours': 0,
                    'last_reset': time.time(),
                    'instances_launched': 0,
                    'successful_sessions': 0,
                    'failed_sessions': 0
                })
                
                # Add active sessions
                active_sessions = [s for s in self.active_sessions.values() if s['username'] == username]
                
                account_usage[username] = {
                    'institution': account.get('institution', ''),
                    'usage_stats': stats,
                    'active_sessions': len(active_sessions)
                }
            
            return {
                'total_accounts': len(self.accounts),
                'active_accounts': len([a for a in self.accounts if a.get('status') == 'active']),
                'total_active_sessions': len(self.active_sessions),
                'account_usage': account_usage
            }
    
    async def check_academic_eligibility(self, email: str, institution: str) -> Dict[str, Any]:
        """Check if an email is eligible for academic access"""
        # In a real implementation, this would verify academic status
        # This is a simplified simulation
        
        # Check if email has academic domain
        academic_domains = [
            '.edu', '.ac.uk', '.ac.jp', '.ac.nz', '.ac.za', '.edu.au', 
            '.edu.sg', '.edu.cn', '.edu.hk', '.edu.tw', '.edu.in'
        ]
        
        is_academic_email = any(email.endswith(domain) for domain in academic_domains)
        
        # Check if institution is in known academic institutions
        known_institutions = [
            'MIT', 'Stanford', 'Harvard', 'Oxford', 'Cambridge', 'ETH Zurich',
            'University of', 'College of', 'Institute of', 'Polytechnic'
        ]
        
        is_known_institution = any(inst.lower() in institution.lower() for inst in known_institutions)
        
        # Determine eligibility
        is_eligible = is_academic_email or is_known_institution
        
        return {
            'email': email,
            'institution': institution,
            'is_eligible': is_eligible,
            'reason': 'Academic email or known institution' if is_eligible else 'Not an academic email or known institution',
            'verification_required': not is_academic_email
        }
    
    async def get_academic_resources(self) -> Dict[str, Any]:
        """Get information about available academic resources"""
        return {
            'instance_types': [
                {
                    'name': 'gpu_1x_a10',
                    'description': 'NVIDIA A10 GPU (24GB VRAM)',
                    'vcpus': 8,
                    'memory_gb': 64,
                    'storage_gb': 200,
                    'academic_price_per_hour': 0.60
                },
                {
                    'name': 'gpu_1x_rtx6000',
                    'description': 'NVIDIA RTX 6000 GPU (24GB VRAM)',
                    'vcpus': 8,
                    'memory_gb': 64,
                    'storage_gb': 200,
                    'academic_price_per_hour': 0.70
                },
                {
                    'name': 'gpu_1x_a100',
                    'description': 'NVIDIA A100 GPU (40GB VRAM)',
                    'vcpus': 16,
                    'memory_gb': 128,
                    'storage_gb': 200,
                    'academic_price_per_hour': 1.10
                },
                {
                    'name': 'gpu_1x_a100_80gb',
                    'description': 'NVIDIA A100 GPU (80GB VRAM)',
                    'vcpus': 16,
                    'memory_gb': 128,
                    'storage_gb': 200,
                    'academic_price_per_hour': 1.50
                }
            ],
            'regions': [
                'us-east-1',
                'us-west-1',
                'eu-west-1',
                'ap-southeast-1'
            ],
            'academic_program': {
                'monthly_credits': 100,
                'eligibility': 'Faculty, researchers, and students at accredited academic institutions',
                'application_url': 'https://lambdalabs.com/academic',
                'verification_process': 'Email verification or institution ID'
            }
        }
    
    async def get_academic_access_techniques(self) -> Dict[str, Any]:
        """Get techniques for maximizing academic access to cloud resources"""
        return {
            'techniques': [
                {
                    'name': 'Institutional Email Registration',
                    'description': 'Register using your .edu or institutional email address for automatic verification',
                    'effectiveness': 'High',
                    'platforms': ['Lambda Labs', 'Google Cloud', 'AWS', 'Azure']
                },
                {
                    'name': 'GitHub Student Developer Pack',
                    'description': 'Register for GitHub Student Developer Pack to get free credits on multiple cloud platforms',
                    'effectiveness': 'High',
                    'platforms': ['DigitalOcean', 'Microsoft Azure', 'Heroku', 'JetBrains']
                },
                {
                    'name': 'Research Grant Applications',
                    'description': 'Apply for research grants specifically for cloud computing resources',
                    'effectiveness': 'Medium',
                    'platforms': ['AWS', 'Google Cloud', 'Azure']
                },
                {
                    'name': 'Open Source Contributions',
                    'description': 'Contribute to open source projects to get access to sponsored resources',
                    'effectiveness': 'Medium',
                    'platforms': ['Various']
                },
                {
                    'name': 'Academic Partnerships',
                    'description': 'Establish partnerships between your institution and cloud providers',
                    'effectiveness': 'High',
                    'platforms': ['All major providers']
                },
                {
                    'name': 'Beta Testing Programs',
                    'description': 'Join beta testing programs for new cloud services to get free access',
                    'effectiveness': 'Medium',
                    'platforms': ['Various']
                },
                {
                    'name': 'Hackathon Sponsorships',
                    'description': 'Organize or participate in hackathons sponsored by cloud providers',
                    'effectiveness': 'Medium',
                    'platforms': ['AWS', 'Google Cloud', 'Azure']
                },
                {
                    'name': 'Research Paper Submissions',
                    'description': 'Submit research papers that utilize specific cloud platforms for potential sponsorship',
                    'effectiveness': 'Medium',
                    'platforms': ['Various']
                }
            ],
            'resource_optimization': [
                {
                    'name': 'Spot Instance Usage',
                    'description': 'Use spot/preemptible instances for non-critical workloads to reduce costs',
                    'savings': 'Up to 90%'
                },
                {
                    'name': 'Scheduled Shutdowns',
                    'description': 'Automatically shut down resources during non-working hours',
                    'savings': 'Up to 70%'
                },
                {
                    'name': 'Resource Right-sizing',
                    'description': 'Use only the resources you need for each specific task',
                    'savings': 'Up to 40%'
                },
                {
                    'name': 'Checkpoint and Resume',
                    'description': 'Save checkpoints of work to resume later, allowing for shorter session times',
                    'savings': 'Up to 50%'
                }
            ]
        }
