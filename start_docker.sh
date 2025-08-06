#!/bin/bash

# MTechCreditScoreVFL Docker Startup Script
# This script helps you start, stop, and manage the Docker services

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Function to check if Docker is running
check_docker() {
    if ! docker info > /dev/null 2>&1; then
        print_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    print_success "Docker is running"
}

# Function to check if docker-compose is available
check_docker_compose() {
    if ! command -v docker-compose &> /dev/null; then
        print_error "docker-compose is not installed. Please install it and try again."
        exit 1
    fi
    print_success "docker-compose is available"
}

# Function to start services
start_services() {
    print_status "Starting MTechCreditScoreVFL services..."
    
    # Check prerequisites
    check_docker
    check_docker_compose
    
    # Check if .env file exists
    if [ ! -f .env ]; then
        print_warning ".env file not found. Creating one with placeholder..."
        echo "# Environment variables for MTechCreditScoreVFL" > .env
        echo "# Add your OpenAI API key here if needed:" >> .env
        echo "# OPENAI_API_KEY=your_api_key_here" >> .env
    fi
    
    # Build and start services
    print_status "Building and starting services..."
    docker-compose up --build -d
    
    print_success "Services started successfully!"
    print_status "Waiting for services to be ready..."
    
    # Wait for services to be ready
    sleep 10
    
    # Check if services are running
    if docker-compose ps | grep -q "Up"; then
        print_success "All services are running!"
        echo ""
        echo "🌐 Access your applications:"
        echo "   📊 Streamlit UI: http://localhost:8501"
        echo "   🔌 Flask API: http://localhost:5001"
        echo ""
        echo "📋 Useful commands:"
        echo "   View logs: docker-compose logs -f"
        echo "   Stop services: docker-compose down"
        echo "   Restart services: docker-compose restart"
    else
        print_error "Some services failed to start. Check logs with: docker-compose logs"
        exit 1
    fi
}

# Function to stop services
stop_services() {
    print_status "Stopping MTechCreditScoreVFL services..."
    docker-compose down
    print_success "Services stopped"
}

# Function to restart services
restart_services() {
    print_status "Restarting MTechCreditScoreVFL services..."
    docker-compose restart
    print_success "Services restarted"
}

# Function to view logs
view_logs() {
    print_status "Showing logs (press Ctrl+C to exit)..."
    docker-compose logs -f
}

# Function to show status
show_status() {
    print_status "Service status:"
    docker-compose ps
}

# Function to clean up
cleanup() {
    print_status "Cleaning up Docker resources..."
    docker-compose down -v --rmi all
    print_success "Cleanup completed"
}

# Function to show help
show_help() {
    echo "MTechCreditScoreVFL Docker Management Script"
    echo ""
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  start     Start the services (default)"
    echo "  stop      Stop the services"
    echo "  restart   Restart the services"
    echo "  logs      View service logs"
    echo "  status    Show service status"
    echo "  cleanup   Stop and remove all containers, volumes, and images"
    echo "  help      Show this help message"
    echo ""
    echo "Examples:"
    echo "  $0 start     # Start services"
    echo "  $0 logs      # View logs"
    echo "  $0 stop      # Stop services"
}

# Main script logic
case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        view_logs
        ;;
    status)
        show_status
        ;;
    cleanup)
        cleanup
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        print_error "Unknown command: $1"
        show_help
        exit 1
        ;;
esac 